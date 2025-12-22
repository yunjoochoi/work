- Page 1 -

![Image](251204_lecture8_llm_agent.pdf__chunk_0000/page_0001/pictures_0.png)

- Page 2 -

## · Attendance

- https://docs.google.com/spreadsheets/d/112HaQ78-H2umGd\_-MDMkflK79wkcr1Ki1Axs8H-isFU/edit?usp=sharing

## Announcements

- Page 3 -

## Introduction: Multi-agent system

- Multi-agent system (or Agentic AI) is one of the hottest topic in AI
- In terms of market size

![Image](251204_lecture8_llm_agent.pdf__chunk_0000/page_0003/pictures_1.png)

- Page 4 -

## Introduction: Multi-agent system

- Multi-agent system (or Agentic AI) is one of the hottest topic in AI
- In terms of market size, research field [1,2]

![Image](251204_lecture8_llm_agent.pdf__chunk_0000/page_0004/pictures_2.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0000/page_0004/pictures_3.png)

- Page 5 -

## Introduction: Multi-agent system

- Multi-agent system?
- [1]: 'multiple intelligent agents' + 'impossible for individual agent'

- Page 6 -

## Introduction: Multi-agent system

- Multi-agent system?
- [1]: 'multiple intelligent agents ' + 'impossible for individual agent' + ' advance of LLMs '

(Perception, reasoning, action) + improving

![Image](251204_lecture8_llm_agent.pdf__chunk_0000/page_0006/pictures_4.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0000/page_0006/pictures_5.png)

- Page 7 -

## Key Concepts in LLM Agents

- Then, what are important for LLM agents?
- LLM + (1) Planning &amp; Action + (2) Tools + (3) Memory

![Image](251204_lecture8_llm_agent.pdf__chunk_0000/page_0007/pictures_6.png)

- Page 8 -

## Key Concepts in LLM Agents

- Then, what are important for LLM agents?
- LLM + (1) Planning &amp; Action + (2) Tools + (3) Memory

Not simply improved by just scaling-up model &amp; data size

![Image](251204_lecture8_llm_agent.pdf__chunk_0000/page_0008/pictures_7.png)

- Page 9 -

- Planning &amp; Action
- Tool using
- Memory

## Contents

- Page 10 -

- Planning &amp; Action
- Tool using
- Memory

## Contents

- Page 11 -

## Planning &amp; Action: ReAct

- ReAct [1] : integrating reasoning and acting within LLM
- By extending action as a combination of task-specific discrete actions and language

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0011/pictures_0.png)

- Page 12 -

## Planning &amp; Action: ReAct

- ReAct [1] : integrating reasoning and acting within LLM
- By extending action as a combination of task-specific discrete actions and language

- Page 13 -

## Planning &amp; Action: ReAct

- ReAct [1] : integrating reasoning and acting within LLM
- By extending action as a combination of task-specific discrete actions and language
- It is implemented by few-shot prompting

- Page 14 -

## Planning &amp; Action: ReAct

- Experimental result
- ReAct → CoT-SC: ReAct fails to return an answer within given steps, back off to CoT-SC
- CoT-SC → ReAct: when # of voting among n paths occurs less than n/2, back off to ReAct

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0014/pictures_1.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0014/pictures_2.png)

- Page 15 -

## Planning &amp; Action: Reflexion

- Reflexion [1] : reinforce LLM agents through linguistic feedback ( self-reflection )

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0015/pictures_3.png)

- Page 16 -

## Planning &amp; Action: Reflexion

- Reflexion [1] : reinforce LLM agents through linguistic feedback ( self-reflection )

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0016/pictures_4.png)

- Page 17 -

## Planning &amp; Action: Reflexion

- Reflexion [1] : reinforce LLM agents through linguistic feedback ( self-reflection )

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0017/pictures_5.png)

- Page 18 -

## Planning &amp; Action: Reflexion

- Reflexion [1] : reinforce LLM agents through linguistic feedback ( self-reflection )

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0018/pictures_6.png)

- Page 19 -

## Planning &amp; Action: Reflexion

- Reflexion [1] : reinforce LLM agents through linguistic feedback ( self-reflection )

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0019/pictures_7.png)

- Page 20 -

## Planning &amp; Action: Reflexion

- Reflexion [1] : reinforce LLM agents through linguistic feedback ( self-reflection )

![Image](251204_lecture8_llm_agent.pdf__chunk_0001/page_0020/pictures_8.png)

- Page 21 -

## Planning &amp; Action: Reflexion

- Experimental results: Sequential decision making (ALFWorld)

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0021/pictures_0.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0021/pictures_1.png)

- Page 22 -

## Planning &amp; Action: Reflexion

- Experimental results: Reasoning (HotpotQA)

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0022/pictures_2.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0022/pictures_3.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0022/pictures_4.png)

- Page 23 -

## Planning &amp; Action: Plan-and-Solve

- Plan-and-solve [1] : explicit planning before solving problem
- Similar to insert 'let's think step -bystep' for zero -shot chain-of-thought (CoT) [2]

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0023/pictures_5.png)

- Page 24 -

## Planning &amp; Action: Plan-and-Solve

- Plan-and-solve [1] : explicit planning before solving problem
- Similar to insert 'let's think step -bystep' for zero -shot chain-of-thought (CoT) [2]

- Page 25 -

## Planning &amp; Action: Tree of Thoughts

- Tree of Thoughts (ToT) [1] : decomposing problem &amp; search with LLM

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0025/pictures_6.png)

- Page 26 -

## Planning &amp; Action: Tree of Thoughts

- Tree of Thoughts (ToT) [1] : decomposing problem &amp; search with LLM
- 1. Thought decomposition : ToT leverages problem properties to design and decompose intermediate thought steps (via prompting)

- Page 27 -

## Planning &amp; Action: Tree of Thoughts

- Tree of Thoughts (ToT) [1] : decomposing problem &amp; search with LLM
- 2. Thought generation : based on query &amp; previous history, generate new thought
- (1) i.i.d sampling similr to CoT-SC, (2) sequential sampling via prompting

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0027/pictures_7.png)

- Page 28 -

## Planning &amp; Action: Tree of Thoughts

- Tree of Thoughts (ToT) [1] : decomposing problem &amp; search with LLM
- 3. State evaluation : estimating goodness of each state via LLM prompting
- (1) Value: scoring each state (1-10), (2) Vote: among all states in same layer, select one

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0028/pictures_8.png)

- Page 29 -

## Planning &amp; Action: Tree of Thoughts

- Tree of Thoughts (ToT) [1] : decomposing problem &amp; search with LLM
- 4. Search algorithm : how to select next state
- (1) Breadth-first search (BFS) and (2) Depth-first search (DFS)

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0029/pictures_9.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0029/pictures_10.png)

- Page 30 -

## Planning &amp; Action: Tree of Thoughts

- Experimental result: Game of 24

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0030/pictures_11.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0002/page_0030/pictures_12.png)

- Page 31 -

## Planning &amp; Action: Tree of Thoughts

- Experimental result: Creative Writing

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0031/pictures_0.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0031/pictures_1.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0031/pictures_2.png)

- Page 32 -

- Planning &amp; Action
- Tool using
- Memory

## Contents

- Page 33 -

## Tool Use by Recent LLMs

- Recent LLMs have capability of using tools

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0033/pictures_3.png)

- Page 34 -

## Prompting-base Approach

- One straight forward way for tooluse with LLMs is ' in-context learning '
- Example: get weather information [1]

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0034/pictures_4.png)

- Page 35 -

## Prompting-base Approach

- One straight forward way for tooluse with LLMs is ' in-context learning '
- Example: get weather information [1]

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0035/pictures_5.png)

- Page 36 -

## Prompting-base Approach

- Step #1 . Call model with functions defined

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0036/pictures_6.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0036/pictures_7.png)

- Page 37 -

## Prompting-base Approach

- Step #2 . Model decides to call function(s)
- Model returns the name and input argument

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0037/pictures_8.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0037/pictures_9.png)

- Page 38 -

## Prompting-base Approach

- Step #3 . Execute function code
- Parse the model's response and handle function calls.

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0038/pictures_10.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0038/pictures_11.png)

- Page 39 -

## Prompting-base Approach

- Step #4 . Supply model with results
- So, model can incorporate them into its final response.

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0039/pictures_12.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0039/pictures_13.png)

- Page 40 -

## Prompting-base Approach

- Step #5 . Model responds -incorporating the result in its output

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0040/pictures_14.png)

![Image](251204_lecture8_llm_agent.pdf__chunk_0003/page_0040/pictures_15.png)

- Page 41 -

## Prompting-base Approach

- How it works? ' System message '

- Page 42 -

## Prompting-base Approach

- How it works? 'System message' + ' Detailed description '

- Page 43 -

## Prompting-base Approach

- How it works? 'System message' + ' Detailed description '
- However, this approach might be suboptimal
- As only relying on implicit in-context learning by LLM (may be only reliable for large LLM)
- When there are tools frequently called and hence should be handling well

- Page 44 -

## Toolformer: Motivation

- Supervised fine-tuning (SFT) by collecting data is continuously applicable
- Using human annotator or large LLM similar to Self-RAG [1]
- However, it necessarily requires large cost
- Also, it totally relies on intrinsic knowledge of large LLM

- Page 45 -

## Toolformer

- Contribution of Toolformer: Self-supervised learning for tool use
- Self-supervised learning: design task from data without human annotation
- Key idea : Useful tool → Helpful to predict next token

![Image](251204_lecture8_llm_agent.pdf__chunk_0004/page_0045/pictures_0.png)

- Page 46 -

## Toolformer

- Let represent API call for tool use as 𝑐 = (𝑎𝑐 , 𝑖 𝑐 )
- 𝑎𝑐 : name of API, 𝑖 𝑐 : corresponding input
- Then, it can be incorporated as text tokens with special tokens '&lt;API&gt;,&lt;/API&gt;, → '

$$e ( c ) = < A P I > a _ { c } ( i _ { c } ) < / A P I >$$

$$\text {result of API call}$$

- Page 47 -

## Toolformer

- Then, converting pre-training corpus by augmenting API calls
- Step 1 . Sampling API Calls through in-context learning

- Page 48 -

## Toolformer

- Then, converting pre-training corpus by augmenting API calls
- Step 2 . Executing API Calls: for each API call 𝑐 yields output 𝑟

- Page 49 -

## Toolformer

- Then, converting pre-training corpus by augmenting API calls
- Step 3 . Filtering API Calls: let 𝑖 be the position of API call
- Then, considering weighted next token prediction loss

$$L _ { i } ( z ) = - \sum _ { j = i } ^ { n } w _ { j - i } \cdot \log p _ { M } ( x _ { j } \ | \ z , x _ { 1 \colon j - 1 } )$$

- With this, let consider two different losses ( 𝜀 : empty sequence)

$$L _ { i } ^ { + } = L _ { i } ( e ( c _ { i } , r _ { i } ) )$$

with function call and response

$$L _ { i } ^ { - } = \min \left ( L _ { i } ( \varepsilon ) , L _ { i } ( e ( c _ { i } , \varepsilon ) ) \right )$$

$$^ { \text {no function call} } \text { no response}$$

- Page 50 -

## Toolformer

- Then, converting pre-training corpus by augmenting API calls
- Step 3 . Filtering API Calls: let 𝑖 be the position of API call
- With this, let consider two different losses ( 𝜀 : empty sequence)

$$L _ { i } ^ { + } = L _ { i } ( e ( c _ { i } , r _ { i } ) )$$

with function call and response

$$L _ { i } ^ { - } = \min \left ( L _ { i } ( \varepsilon ) , L _ { i } ( \mathbf e ( c _ { i } , \varepsilon ) ) \right )$$

$$^ { \text {no function call} } \text { no response}$$

- Only keep API calls for which 𝐿 𝑖 - -𝐿𝑖 + ≥ 𝜏 holds
- i.e., adding API call and its result when it reduces loss by at least 𝜏 , compared to not doing any API call or obtaining no result from it

- Page 51 -

## Toolformer

- Then, converting pre-training corpus by augmenting API calls
- Step 4 . Model fine-tuning

- Page 52 -

## Toolformer

- Then, converting pre-training corpus by augmenting API calls
- Step 5 . Inference

- Page 53 -

## Toolformer: Results

- Tool use makes significant improvement for various tasks

- Page 54 -

## ToolLLM

- Goal : scalable instruction tuning for tool use capability of open-sourced LLMs
- Focusing on developing data collection pipeline specified for tool use
- By leveraging strong closed LLMs such as ChatGPT

![Image](251204_lecture8_llm_agent.pdf__chunk_0005/page_0054/pictures_0.png)

- Page 55 -

## ToolLLM

- Goal : scalable instruction tuning for tool use capability of open-sourced LLMs
- Focusing on developing data collection pipeline specified for tool use
- By leveraging strong closed LLMs such as ChatGPT

![Image](251204_lecture8_llm_agent.pdf__chunk_0005/page_0055/pictures_1.png)

- Page 56 -

## ToolLLM: API Collection

- By filtering tools from RapidAPI, collecting 3,451 tools with 16,464 APIs
- RapidAPI: API marketplace that connects developers with thousands of real-world APIs

![Image](251204_lecture8_llm_agent.pdf__chunk_0005/page_0056/pictures_2.png)

- Page 57 -

## ToolLLM: Instruction Generation

- Then, using ChatGPT, generate instruction that requires tool use to answer
- Focus on two aspects: (1) diversity &amp; (2) multi-tool usage
- Prompt is composed of (1) description, (2) documentation for APIs, (3) 3 few-shot data
- 12/36 diverse few-shot examples for single-tool/multi-tool setting (randomly sampled)

- Page 58 -

## ToolLLM: Instruction Generation

- Then, using ChatGPT, generate instruction that requires tool use to answer
- With total API set 𝑆𝐴𝑃𝐼 , at each time, 𝑁 APIs 𝑆𝑁 𝑠𝑢𝑏 = {𝐴𝑃𝐼1, … , 𝐴𝑃𝐼𝑁} are sampled
- Then, generating possible instructions ( 𝐼𝑛𝑠𝑡∗ ) that involve APIs 𝑆∗ 𝑠𝑢𝑏 ⊂ 𝑆𝑁 𝑠𝑢𝑏

$$\{ [ \mathbb { S } _ { 1 } ^ { \text {rel} } , \text {Inst} _ { 1 } ] , \cdots , [ \mathbb { S } _ { N ^ { \prime } } ^ { \text {rel} } , \text {Inst} _ { N ^ { \prime } } ] \}$$

- After filtering, nearly 200k qualified pairs are collected
- These instruction and relevant API pairs is utilized to train API retriever

- Page 59 -

## ToolLLM: Instruction Generation

## · Example of prompt: 'Task description'

- Page 60 -

## ToolLLM: Instruction Generation

## · Example of prompt: 'Few -shot examples'

- Page 61 -

## ToolLLM: Solution Path Annotation

- To generate validate action for given 𝐼𝑛𝑠𝑡∗ and APIs, prompting ChatGPT
- Then, ChatGPT will generate sequence of actions {𝑎1 , … , 𝑎𝑁}
- At each round 𝑡 , ChatGPT generates 𝑎𝑡 based on previous interactions

$$\Gamma \, A P I \, r e s p o n s e \, f o r \, a c t i o n$$

- Specifically, 𝑎𝑡 has below format

- Page 62 -

## ToolLLM: Solution Path Annotation

- Specifically, function call feature of ChatGPT is utilized
- Namely, sampled APIs 𝑆𝑁 𝑠𝑢𝑏 are fed as available functions (in system message level)
- To let ChatGPT finish an action sequence, two additional functions are defined
- (1) Finish with final answer &amp; (2) Finish by giving up
- Depth first search is considered to efficiently search and collect data

![Image](251204_lecture8_llm_agent.pdf__chunk_0006/page_0062/pictures_0.png)

- Page 63 -

## ToolLLM: Overview

- Overview at inference time (similar to RAG)
- For a given test instruction,
- 1) Retrieving relevant tools &amp; APIs (retrieval of document in RAG)
- 2) With given APIs, generate proper actions of reasoning &amp; API call (generation in RAG)

![Image](251204_lecture8_llm_agent.pdf__chunk_0006/page_0063/pictures_1.png)

- Page 64 -

## ToolLLM: Experiment - Retriever

- Dense retriever is trained based on BERT (similar to DPR)
- Query: instruction, Passage: API document
- Positive pair: instruction &amp; relevant API, negative: using API by other instruction
- It even outperforms embedding model (Ada) provided by OpenAI

- Page 65 -

## ToolLLM: Experiment - Generation

- TooLLaMA: finetuned LLaMA -7B using instruction-solution pairs
- Two metrics: (1) Pass: answerability by LLM &amp; (2) Win: quality compared to ChatGPT
- Retrieved top-5 APIs are even better than ground-truth 𝑆𝑁 𝑠𝑢𝑏 used to generate data