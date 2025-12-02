"""3-stage LLM Council orchestration."""

from collections import defaultdict
from typing import List, Dict, Any, Tuple
from .llm_client import query_models_parallel, query_model
from .config import (
    LOCAL_DEFAULT_MODEL,
)
from .settings import (
    get_settings,
    DEFAULT_RANKING_PROMPT,
    DEFAULT_CHAIRMAN_PROMPT,
    DEFAULT_TITLE_PROMPT,
)


async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:
    """
    Stage 1: Collect individual responses from all council models.

    Args:
        user_query: The user's question

    Returns:
        List of dicts with 'model' and 'response' keys
    """
    settings = get_settings()
    council_models = settings.get("council_models", [])

    messages = [{"role": "user", "content": user_query}]

    # Query all models in parallel
    responses = await query_models_parallel(council_models, messages)

    # Format results
    stage1_results = []
    for model, response in responses.items():
        if response is not None:  # Only include successful responses
            stage1_results.append({
                "model": model,
                "response": response.get('content', ''),
                "response_time": response.get('response_time')
            })

    return stage1_results


def format_prompt(template: str, values: Dict[str, str]) -> str:
    """Safely format a prompt template with fallback empty strings for missing keys."""
    return template.format_map(defaultdict(str, values))


async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Stage 2: Each model ranks the anonymized responses.

    Args:
        user_query: The original user query
        stage1_results: Results from Stage 1

    Returns:
        Tuple of (rankings list, label_to_model mapping)
    """
    settings = get_settings()
    council_models = settings.get("council_models", [])

    # Create anonymized labels for responses (Response A, Response B, etc.)
    labels = [chr(65 + i) for i in range(len(stage1_results))]  # A, B, C, ...

    # Create mapping from label to model name
    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    # Build the ranking prompt
    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_template = settings.get("ranking_prompt", DEFAULT_RANKING_PROMPT)
    ranking_prompt = format_prompt(
        ranking_template,
        {
            "user_query": user_query,
            "responses_text": responses_text,
        },
    )

    messages = [{"role": "user", "content": ranking_prompt}]

    # Get rankings from all council models in parallel
    responses = await query_models_parallel(council_models, messages)

    # Format results
    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed,
                "response_time": response.get('response_time')
            })

    return stage2_results, label_to_model


async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Stage 3: Chairman synthesizes final response.

    Args:
        user_query: The original user query
        stage1_results: Individual model responses from Stage 1
        stage2_results: Rankings from Stage 2

    Returns:
        Dict with 'model' and 'response' keys
    """
    settings = get_settings()
    chairman_model_used = settings.get("chairman_model", LOCAL_DEFAULT_MODEL)
    provider = settings.get("llm_provider", "openrouter")
    local_default_model = settings.get("local_default_model", LOCAL_DEFAULT_MODEL)

    # Build comprehensive context for chairman
    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking: {result['ranking']}"
        for result in stage2_results
    ])

    chairman_template = settings.get("chairman_prompt", DEFAULT_CHAIRMAN_PROMPT)
    chairman_prompt = format_prompt(
        chairman_template,
        {
            "user_query": user_query,
            "stage1_text": stage1_text,
            "stage2_text": stage2_text,
        },
    )

    messages = [{"role": "user", "content": chairman_prompt}]

    # Query the chairman model
    response = await query_model(chairman_model_used, messages)

    # Fallback to local default if the requested model is missing when using Ollama
    if response is None and provider == "ollama" and chairman_model_used != local_default_model:
        chairman_model_used = local_default_model
        response = await query_model(chairman_model_used, messages)

    if response is None:
        # Fallback if chairman fails
        return {
            "model": chairman_model_used,
            "response": "Error: Unable to generate final synthesis.",
            "response_time": None,
        }

    return {
        "model": chairman_model_used,
        "response": response.get('content', ''),
        "response_time": response.get('response_time'),
    }


def parse_ranking_from_text(ranking_text: str) -> List[str]:
    """
    Parse the FINAL RANKING section from the model's response.

    Args:
        ranking_text: The full text response from the model

    Returns:
        List of ranking labels in order (e.g., ["Response A", "Response B"])
    """
    import re

    # Look for "FINAL RANKING:" section
    if "FINAL RANKING:" in ranking_text:
        # Extract everything after "FINAL RANKING:"
        parts = ranking_text.split("FINAL RANKING:")
        if len(parts) >= 2:
            ranking_section = parts[1]
            # Try to extract numbered list format (e.g., "1. Response A")
            # This pattern looks for: number, period, optional space, "Response X"
            numbered_matches = re.findall(r'\d+\.\s*Response [A-Z]', ranking_section)
            if numbered_matches:
                # Extract just the "Response X" part
                return [re.search(r'Response [A-Z]', m).group() for m in numbered_matches]

            # Fallback: Extract all "Response X" patterns in order
            matches = re.findall(r'Response [A-Z]', ranking_section)
            return matches

    # Fallback: try to find any "Response X" patterns in order
    matches = re.findall(r'Response [A-Z]', ranking_text)
    return matches


def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Calculate aggregate rankings across all models.

    Args:
        stage2_results: Rankings from each model
        label_to_model: Mapping from anonymous labels to model names

    Returns:
        List of dicts with model name and average rank, sorted best to worst
    """
    from collections import defaultdict

    # Track positions for each model
    model_positions = defaultdict(list)

    for ranking in stage2_results:
        ranking_text = ranking['ranking']

        # Parse the ranking from the structured format
        parsed_ranking = parse_ranking_from_text(ranking_text)

        for position, label in enumerate(parsed_ranking, start=1):
            if label in label_to_model:
                model_name = label_to_model[label]
                model_positions[model_name].append(position)

    # Calculate average position for each model
    aggregate = []
    for model, positions in model_positions.items():
        if positions:
            avg_rank = sum(positions) / len(positions)
            aggregate.append({
                "model": model,
                "average_rank": round(avg_rank, 2),
                "rankings_count": len(positions)
            })

    # Sort by average rank (lower is better)
    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate


async def generate_conversation_title(user_query: str) -> str:
    """
    Generate a short title for a conversation based on the first user message.

    Args:
        user_query: The first user message

    Returns:
        A short title (3-5 words)
    """
    settings = get_settings()
    title_model_used = settings.get("title_model", LOCAL_DEFAULT_MODEL)
    provider = settings.get("llm_provider", "openrouter")
    local_default_model = settings.get("local_default_model", LOCAL_DEFAULT_MODEL)

    title_template = settings.get("title_prompt", DEFAULT_TITLE_PROMPT)
    title_prompt = format_prompt(
        title_template,
        {
            "user_query": user_query,
        },
    )

    messages = [{"role": "user", "content": title_prompt}]

    # Use the configured model for title generation
    response = await query_model(title_model_used, messages, timeout=30.0)

    # If the preferred model is unavailable on Ollama, fall back to the configured local default
    if response is None and provider == "ollama" and title_model_used != local_default_model:
        title_model_used = local_default_model
        response = await query_model(title_model_used, messages, timeout=30.0)

    if response is None:
        # Fallback to a generic title
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    # Clean up the title - remove quotes, limit length
    title = title.strip('"\'')

    # Truncate if too long
    if len(title) > 50:
        title = title[:47] + "..."

    return title


async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:
    """
    Run the complete 3-stage council process.

    Args:
        user_query: The user's question

    Returns:
        Tuple of (stage1_results, stage2_results, stage3_result, metadata)
    """
    # Stage 1: Collect individual responses
    stage1_results = await stage1_collect_responses(user_query)

    # If no models responded successfully, return error
    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond. Please try again."
        }, {}

    # Stage 2: Collect rankings
    stage2_results, label_to_model = await stage2_collect_rankings(user_query, stage1_results)

    # Calculate aggregate rankings
    aggregate_rankings = calculate_aggregate_rankings(stage2_results, label_to_model)

    # Stage 3: Synthesize final answer
    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )

    # Prepare metadata
    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, stage3_result, metadata
