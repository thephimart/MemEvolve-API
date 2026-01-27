"""
Streaming utilities for handling Server-Sent Events and streaming responses.
"""

import json
from typing import Union


def extract_final_from_stream(response_str: str) -> Union[str, bytes]:
    """Extract the final complete response from a streaming SSE response."""
    lines = response_str.strip().split('\n')
    accumulated_content = ""
    accumulated_reasoning = ""
    final_data = None

    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data_content = line[6:].strip()  # Remove 'data: ' prefix
            if data_content and data_content != '[DONE]':
                try:
                    parsed = json.loads(data_content)
                    choice = parsed.get('choices', [{}])[0]

                    # Accumulate content from all chunks
                    delta = choice.get('delta', {})
                    if 'content' in delta and delta['content'] is not None:
                        accumulated_content += delta['content']
                    if 'reasoning_content' in delta and delta['reasoning_content'] is not None:
                        accumulated_reasoning += delta['reasoning_content']

                    # Look for the final chunk (finish_reason is not null)
                    if choice.get('finish_reason') is not None:
                        # Create a complete response with accumulated content
                        final_data = {
                            "choices": [{
                                "finish_reason": choice.get('finish_reason'),
                                "index": choice.get('index', 0),
                                "delta": {
                                    "role": "assistant",
                                    "content": accumulated_content,
                                    "reasoning_content": accumulated_reasoning
                                }
                            }],
                            "created": parsed.get('created', 0),
                            "id": parsed.get('id', ''),
                            "model": parsed.get('model', ''),
                            "object": "chat.completion.chunk"
                        }
                        break
                    # Keep track of the latest complete chunk for fallback
                    final_data = data_content
                except json.JSONDecodeError:
                    continue

    if isinstance(final_data, dict):
        # Return the constructed complete response
        return json.dumps(final_data)
    elif final_data:
        return final_data
    else:
        # Fallback: return the last data line if no finish_reason found
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('data: ') and line != 'data: [DONE]':
                return line[6:].strip()

    return '{"error": "could_not_extract_final_response"}'
