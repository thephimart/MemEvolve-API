"""
Streaming utilities for handling Server-Sent Events and streaming responses.
"""

import json
from typing import Union


def extract_final_from_stream(response_str: str) -> Union[str, bytes]:
    """Extract the final complete response from a streaming SSE response."""
    lines = response_str.strip().split('\n')
    final_data = None

    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data_content = line[6:].strip()  # Remove 'data: ' prefix
            if data_content and data_content != '[DONE]':
                try:
                    parsed = json.loads(data_content)
                    # Look for the final chunk (finish_reason is not null)
                    if parsed.get('choices', [{}])[0].get('finish_reason') is not None:
                        final_data = data_content
                        break
                    # Keep track of the latest complete chunk
                    final_data = data_content
                except json.JSONDecodeError:
                    continue

    if final_data:
        return final_data
    else:
        # Fallback: return the last data line if no finish_reason found
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('data: ') and line != 'data: [DONE]':
                return line[6:].strip()

    return '{"error": "could_not_extract_final_response"}'