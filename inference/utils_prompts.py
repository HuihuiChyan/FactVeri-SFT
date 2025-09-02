CLS_VERDICT_PROMPT = """You are an expert fact-checking assistant. Your task is to determine whether the answer is factually correct or not.

Question: {question}
Answer: {answer}
Retrieved Reference Information: {formatted_facts}

Based on the question, answer, and the reference information retrieved before, determine if the answer is factually correct or not."""

CLS_VERDICT_PROMPT_NAIVE = """You are an expert fact-checking assistant. Your task is to determine whether the answer is factually correct or not.

Question: {question}
Answer: {answer}

Based on the question, answer, determine if the answer is factually correct or not."""