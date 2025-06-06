import json

def decode_openai_output(chunk_list):
    output_list = []
    cur_output = ""  # Buffer for accumulating complete output
    json_buffer = ""  # Buffer for building complete JSON strings
    
    for chunk in chunk_list:
        try:
            # Convert bytes to string and add to buffer
            if isinstance(chunk, bytes):
                chunk_str = chunk.decode("utf-8")
            else:
                chunk_str = chunk
                
            json_buffer += chunk_str

            # Process all complete JSON messages in the buffer
            while True:
                # Look for complete data: message
                start_idx = json_buffer.find('data:')
                if start_idx == -1:
                    break
                    
                # Find the end of the current message (next 'data:' or end of string)
                next_data_idx = json_buffer.find('data:', start_idx + 5)
                if next_data_idx == -1:
                    # If no next 'data:', check if we have a complete message
                    if '\n\n' in json_buffer[start_idx:] or json_buffer.strip().endswith('}'):
                        message_end = len(json_buffer)
                    else:
                        break  # Incomplete message, wait for more chunks
                else:
                    message_end = next_data_idx

                # Extract and process the complete message
                message = json_buffer[start_idx:message_end].strip()
                if message.startswith('data:'):
                    content = message[5:].strip()
                    if content == '[DONE]':
                        if cur_output:
                            output_list.append(cur_output.strip())
                        json_buffer = json_buffer[message_end:]
                        continue
                        
                    try:
                        output_data = json.loads(content)
                        new_text = output_data["choices"][0]["text"]
                        cur_output += new_text
                        output_list.append(cur_output.strip())
                    except (json.JSONDecodeError, KeyError):
                        # Skip malformed JSON
                        pass
                        
                json_buffer = json_buffer[message_end:]
                
        except Exception as e:
            # Continue processing even if a chunk fails
            continue

    # Handle any remaining complete content in buffer
    if json_buffer.strip().startswith('data:'):
        content = json_buffer[5:].strip()
        if content and content != '[DONE]':
            try:
                output_data = json.loads(content)
                new_text = output_data["choices"][0]["text"]
                cur_output += new_text
                output_list.append(cur_output.strip())
            except (json.JSONDecodeError, KeyError):
                pass

    return output_list

# Test with the provided data
if __name__ == "__main__":
    test_chunks = [
        b'data: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":" ","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":"1","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":".","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":"5","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata',
        b': {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":" ","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":"1","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":".","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":"5","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"',
        b'id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":" ","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":"1","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":".","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":"5","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":',
        b'"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":" ","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":"1","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":".","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\ndata: {"id":"29849206YX2due8TivYYaxNKwiQ3o/cmpl-96083d312c7047b59c16c78857f6f05d","object":"text_completion","created":1743164100,"model":"deepseek","choices":[{"index":0,"text":"5","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\n'
    ]
    
    result = decode_openai_output(test_chunks)
    print(result)