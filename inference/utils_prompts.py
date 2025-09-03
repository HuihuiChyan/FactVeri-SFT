CLS_VERDICT_PROMPT = """You are an expert fact-checking assistant. Your task is to determine whether the answer is factually correct or not.

Question: {question}
Answer: {answer}
Retrieved Reference Information: {formatted_facts}

Based on the question, answer, and the reference information retrieved before, determine if the answer is factually correct or not."""

CLS_VERDICT_PROMPT_NAIVE = """You are an expert fact-checking assistant. Your task is to determine whether the answer is factually correct or not.

Question: {question}
Answer: {answer}

Based on the question, answer, determine if the answer is factually correct or not."""

POINTWISE_VERDICT_PROMPT = """Based on the preceding conversation, your task is to first summarize the information you have retrieved, and then determine if the provided answer is factually correct for the question.

Question: {question}
Answer: {answer}

Please perform the following three steps:
1. Summarize the useful information from the retrieval results in the format: '**Useful Information**: <facts> Fact 1; Fact 2; Fact 3 </facts>'. If no useful information was found, use '**Useful Information**: <facts> No useful information retrieved. </facts>'.
2. Based on summarized information, provide your reasoning for the verdict in the format: '**Verdict Reasoning**: <reasoning> Your reasoning process... </reasoning>'.
3. After reasoning, state your final verdict in the format: '**Final Verdict**: <verdict> Correct/Incorrect/Intermediate </verdict>'."""