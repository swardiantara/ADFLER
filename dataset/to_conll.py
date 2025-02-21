import string


def to_conll(line):
    m_id, message, entity = line
    entity = 'Event' if entity == 'E' else "NonEvent"
    tokens = message.split()
    n_token = len(tokens)
    tag_list = []
    token_list = []
    # Single token message
    if n_token == 1:
        print(f'single token: {tokens}')
        [token] = tokens
        if token[-1] in string.punctuation:
            last_t = token[:-2]
            punct = token[-1]
            token_list.append(last_t)
            token_list.append(punct)
            tag_list.append(f'S-{entity}')
            tag_list.append('O')
        else:
            tag_list.append(f'S-{entity}')
        print(f'token_list: {token_list}')
        print(f'tag_list: {tag_list}')
    else:
        for i, token in enumerate(tokens):
            print(i)
            if i == 0: ## first token, assign B-tag
                token_list.append(token)
                tag_list.append(f'B-{entity}')
            elif i == n_token - 1: ## last token, process the same way as single token
                if token[-1] in string.punctuation:
                    last_t = token[:len(token)-1]
                    punct = token[-1]
                    token_list.append(last_t)
                    token_list.append(punct)
                    tag_list.append(f'E-{entity}')
                    tag_list.append('O')
            else:
                token_list.append(token)
                tag_list.append(f'I-{entity}')

    
    return [m_id] * len(token_list), token_list, tag_list


def main():
    # Read from file
    # file = open("pybsd_output.txt", "r")
    prev_mid = 1
    with open("pybsd_output.txt", "r") as f_in:
        log_records = f_in.readlines()
        for line in log_records:
            line = [element.strip() for element in line.split('##')]
            current_mid, _, _ = line
            m_ids, tokens, tags = to_conll(line)
            print(f'm_ids: {(m_ids)}, \ntokens: {(tokens)}, \ntags: {(tags)}')
            print(f'm_ids: {len(m_ids)}, tokens: {len(tokens)}, tags: {len(tags)}')
            assert len(m_ids) == len(tokens) and len(m_ids) == len(tags)
            
            with open('dataset_conll.txt', "a") as f_out:
                for i in range(0, len(m_ids)):
                    if(current_mid == prev_mid):
                        print("{} X X {}".format(tokens[i], tags[i]), file=f_out)
                    else: # Newline for sentence separator
                        print("\n{} X X {}".format(tokens[i], tags[i]), file=f_out)
                        prev_mid = current_mid


if __name__ == "__main__":
    main()