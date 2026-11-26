
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
import os
import re

def policy_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower())

def build_lexical_retriever(documents):
    if not documents:
        raise ValueError("At least one document is required to build BM25 retrieval.")
    return BM25Retriever.from_documents(
        documents,
        preprocess_func=policy_tokens,
        k=min(20, len(documents)),
    )

# Retrieve the top-k relevant chunks from dense and lexical search
def retrieve(query: str, k: int = 4, vectorstore=None, lexical_retriever=None) -> list[str]:
    if vectorstore is None or lexical_retriever is None:
        raise ValueError("Both the vector store and BM25 retriever are required.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    if vectorstore.index.ntotal < 1:
        raise ValueError("The vector store contains no documents.")

    query_lower = query.lower()
    property_codes = re.findall(r"\bASH-[A-Z]{3}\b", query.upper())
    rate_terms = [
        term for term in ("advance purchase", "flexible", "member", "promotional")
        if term in query_lower
    ]
    property_fields = []
    if any(term in query_lower for term in ("check in", "check-in", "checkin", "check out", "check-out", "checkout", "early arrival", "late departure", "arrival time", "departure time", "be out")):
        property_fields.append("check-in / out")
    if "parking" in query_lower:
        property_fields.append("parking")
    if any(term in query_lower for term in ("pet", "dog", "service animal", "service dog")):
        property_fields.extend(("pets", "pet fee"))
    if "breakfast" in query_lower:
        property_fields.append("breakfast")
    expansion_terms = []
    if property_fields and "check-in / out" in property_fields:
        expansion_terms.append("standard check-in check-out local property time early arrival late departure room readiness subject to availability approval")
    if "parking" in property_fields:
        expansion_terms.append("parking type nightly charge not guaranteed parking charges separate room rate confirmation live property profile")
    if "breakfast" in property_fields:
        expansion_terms.append("breakfast rate plan age-based pricing included breakfast not a group-wide guarantee")
    if any(term in query_lower for term in ("service dog", "service animal")):
        expansion_terms.append("ADA permitted questions work or task trained to perform no certificate")
    if any(term in query_lower for term in ("smok", "vape")):
        expansion_terms.append("smoking vaping remediation up to $250 credible evidence guest agreement")
    if any(term in query_lower for term in ("noise", "noisy", "loud", "disturb")):
        expansion_terms.append("quiet hours 10:00 PM 7:00 AM document complaint escalate disturbance immediate threat violence call 911 emergency services")
    if any(term in query_lower for term in ("pending", "incidental", "hold", "charged twice")):
        expansion_terms.append("temporary authorization pending hold not a posted charge release card issuer")
    if any(term in query_lower for term in ("cancel", "cancellation", "refund")):
        expansion_terms.append("rate plan cancellation deadline local property time one day before arrival")
    if "advance purchase" in rate_terms and any(term in query_lower for term in ("cancel", "cancellation", "refund")):
        expansion_terms.append("Advance Purchase refunds verification confirmation terms payment capture manager approval reason for exception")
    if "refund" in query_lower and any(term in query_lower for term in ("late", "after the deadline")):
        expansion_terms.append("manager exception documented service failure duplicate booking verified system error approved service recovery")
    if any(term in query_lower for term in ("pool", "cabana", "amenity", "amenities")):
        expansion_terms.append("pool cabana amenities property profile availability reservations not in the matrix live property profile")
    retrieval_query = " ".join([query] + expansion_terms)

    candidate_k = min(max(3 * k, 12), vectorstore.index.ntotal)
    dense_results = vectorstore.similarity_search(retrieval_query, k=candidate_k)
    keyword_results = lexical_retriever.invoke(retrieval_query)
    scope_results = [
        document
        for code in property_codes
        for document in lexical_retriever.invoke(code)
    ]

    scores = {}
    documents = {}
    for weight, ranked_docs in ((1.0, dense_results), (2.0, keyword_results), (1.0, scope_results)):
        for rank, document in enumerate(ranked_docs):
            key = document.page_content
            scores[key] = scores.get(key, 0.0) + weight / (rank + 1)
            documents[key] = document

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    priority_keys = []
    if property_codes:
        requested_codes = set(property_codes)
        has_exact_property = any(
            requested_codes.intersection(re.findall(r"\bASH-[A-Z]{3}\b", documents[key].page_content.upper()))
            for key in ranked_keys
        )
        if has_exact_property:
            ranked_keys = [
                key for key in ranked_keys
                if not set(re.findall(r"\bASH-[A-Z]{3}\b", documents[key].page_content.upper()))
                or requested_codes.intersection(re.findall(r"\bASH-[A-Z]{3}\b", documents[key].page_content.upper()))
            ]
        if property_fields:
            relevant_property_keys = [
                key for key in ranked_keys
                if requested_codes.intersection(re.findall(r"\bASH-[A-Z]{3}\b", documents[key].page_content.upper()))
                and any(field in documents[key].page_content.lower() for field in property_fields)
            ]
            if relevant_property_keys:
                ranked_keys = [
                    key for key in ranked_keys
                    if not requested_codes.intersection(re.findall(r"\bASH-[A-Z]{3}\b", documents[key].page_content.upper()))
                    or key in relevant_property_keys
                ]
            else:
                ranked_keys = [
                    key for key in ranked_keys
                    if not requested_codes.intersection(re.findall(r"\bASH-[A-Z]{3}\b", documents[key].page_content.upper()))
                ]
    if rate_terms:
        scoped_keys = []
        for key in ranked_keys:
            plans = {plan.lower() for plan in re.findall(
                r"\brate plan:\s*(advance purchase|flexible|member|promotional)\b",
                documents[key].page_content,
                re.IGNORECASE,
            )}
            if not plans or plans.intersection(rate_terms):
                scoped_keys.append(key)
        ranked_keys = scoped_keys
    for code in property_codes:
        matches = [key for key in ranked_keys if code.lower() in documents[key].page_content.lower()]
        field_matches = [
            key for key in matches
            if any(field in documents[key].page_content.lower() for field in property_fields)
        ]
        for match in (field_matches if property_fields else matches[:1]):
            if match not in priority_keys:
                priority_keys.append(match)

    intent_requirements = []
    if any(term in query_lower for term in ("service dog", "service animal")):
        intent_requirements.append(("service animal", "work or task"))
        intent_requirements.append(("certificate", "medical record"))
    if property_fields and "check-in / out" in property_fields:
        intent_requirements.extend((("early arrival", "availability"), ("late departure", "availability")))
    if "parking" in property_fields:
        intent_requirements.append(("parking charges", "room rate"))
    if "breakfast" in property_fields:
        intent_requirements.append(("breakfast", "rate plan"))
    if any(term in query_lower for term in ("smok", "vape")):
        intent_requirements.append(("$250", "smoking"))
    if any(term in query_lower for term in ("noise", "noisy", "loud", "disturb")):
        intent_requirements.append(("quiet hours", "document"))
        intent_requirements.append(("call 911", "immediate threat"))
    if any(term in query_lower for term in ("pending", "incidental", "hold", "charged twice")):
        intent_requirements.append(("pending", "issuer"))
    if "refund" in query_lower and any(term in query_lower for term in ("late", "after the deadline")):
        intent_requirements.append(("manager", "exception"))
        intent_requirements.append(("documented service failure", "duplicate booking"))
    if "advance purchase" in rate_terms and any(term in query_lower for term in ("cancel", "cancellation", "refund")):
        intent_requirements.append(("advance purchase refunds", "manager approval"))
    if any(term in query_lower for term in ("pool", "cabana", "amenity", "amenities")):
        intent_requirements.extend((("fitness and pool", "property profile"), ("not in the matrix", "live property profile")))
    for required_terms in intent_requirements:
        match = next(
            (key for key in ranked_keys if all(term in documents[key].page_content.lower() for term in required_terms)),
            None,
        )
        if match is not None and match not in priority_keys:
            priority_keys.append(match)

    for term in rate_terms:
        matches = [key for key in ranked_keys if term in documents[key].page_content.lower()]
        rate_plan_matches = [
            key for key in matches
            if re.search(rf"\brate plan:\s*{re.escape(term)}\b", documents[key].page_content, re.IGNORECASE)
        ]
        cancellation_matches = [
            key for key in matches
            if re.search(r"\b(cancel|cancellation|refund|change|charge)\b", documents[key].page_content, re.IGNORECASE)
        ]
        preferred_matches = rate_plan_matches or cancellation_matches or matches
        match = next((key for key in ranked_keys if key in preferred_matches), None)
        if match is not None and match not in priority_keys:
            priority_keys.append(match)
    results = priority_keys[:k] if priority_keys else ranked_keys[:k]
    return [
        f"Content:\n{documents[key].page_content}\n\nMetadata:\n{documents[key].metadata}"
        for key in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are AuroraStay's hotel policy assistant. Answer guest and staff questions using only the retrieved policy information below.
    Distinguish company-wide baselines from property- and rate-specific terms. Never apply one property's details to another.
    If information is missing or ambiguous, state what is not specified and recommend checking the reservation confirmation or live property profile. Do not guess.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely.
    4. Do not omit important conditions, exceptions, fees, deadlines, property-specific restrictions, or approval requirements.
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly and ensure the response addresses what was asked.
    8. Keep the answer concise, clear, and easy to understand.
    9. Do not invent rates, fees, availability, amenities, cancellation terms, or exceptions.
    10. If the information is insufficient, say:
        "The available policy documents do not specify this. Please check the reservation confirmation or the property's live profile."
        If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None,
    lexical_retriever=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(
        query=query,
        k=k,
        vectorstore=vectorstore,
        lexical_retriever=lexical_retriever,
    )

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
