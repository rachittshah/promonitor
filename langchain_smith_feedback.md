```python
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.evaluation import StringEvaluator
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import re

class CustomFeedbackEvaluator(StringEvaluator):
    """An LLM-based custom feedback evaluator."""

    def __init__(self):
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        template = """You are an expert professor specialized in grading students' answers to questions.
        You are grading the following question:
        {query}
        Here is the real answer:
        {answer}
        You are grading the following predicted answer:
        {result}
        Respond with a score from 0 to 100 based on the following criteria:
        - Relevance to the question
        - Accuracy of the answer
        - Clarity of the answer
        - Depth of the answer
        Grade:
        """

        self.eval_chain = LLMChain.from_string(llm=llm, template=template)

    @property
    def requires_input(self) -> bool:
        return True

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "custom_feedback"

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        evaluator_result = self.eval_chain(
            dict(input=input, prediction=prediction, answer=reference), **kwargs
        )
        reasoning, score = evaluator_result["text"].split("\n", maxsplit=1)
        score = re.search(r"\d+", score).group(0)
        if score is not None:
            score = float(score.strip()) / 100.0
        return {"score": score, "reasoning": reasoning.strip()}
```

```python
client = Client()
evaluation_config = RunEvalConfig(
    custom_evaluators = [CustomFeedbackEvaluator()],
)

run_on_dataset(
    dataset_name="<my_dataset_name>",
    llm_or_chain_factory=<llm or function constructing chain>,
    client=client,
    evaluation=evaluation_config,
    project_name="<the name to assign to this test project>",
)
```t's current feedback mechanism primarily relies on binary feedback (thumbs up/thumbs down). While this provides a simple way to gauge user satisfaction, it lacks the granularity and specificity needed to understand the nuances of user experience and model performance. To address this, we propose a more comprehensive feedback mechanism that incorporates multiple dimensions of evaluation.

## 1. Multi-Dimensional Scoring

Instead of a binary score, we can introduce a Likert scale (1-5 or 1-7) for different dimensions of the interaction. These dimensions could include:

- Relevance: Does the output align with the input prompt?
- Coherence: Is the output logically consistent and understandable?
- Creativity: Does the output demonstrate novel or imaginative thinking?
- Correctness: Is the output factually accurate (if applicable)?
- Helpfulness: Was the output useful to the user?

```python
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

evaluation_config = RunEvalConfig(
    custom_evaluators=[
        RelevanceEvaluator(),
        CoherenceEvaluator(),
        CreativityEvaluator(),
        CorrectnessEvaluator(),
        HelpfulnessEvaluator(),
    ],
)
run_on_dataset(
    client=client,
    dataset_name="<my_dataset_name>",
    llm_or_chain_factory=<llm or function constructing chain>,
    evaluation=evaluation_config,
)
```

Users can rate each dimension after an interaction, providing a more detailed picture of the model's performance.

## 2. Open-Ended Feedback

In addition to the Likert scale ratings, users should have the option to provide open-ended feedback. This can capture nuances that may not be reflected in the ratings and provide valuable insights for improving the model.

```python
client.create_feedback(run_id, "open_ended_feedback", comment="The output was creative but not very relevant to my prompt.")
```

## 3. Error Classification

When users encounter issues with the model's output, they should have the option to classify the type of error. This could include categories like "Factually Incorrect", "Unintelligible", "Offensive Content", etc. This can help identify common issues and prioritize improvements.

```python
client.create_feedback(run_id, "error_classification", value="Factually Incorrect")
```

## 4. Contextual Feedback

Feedback should be collected not just for the overall interaction, but also for specific parts of the output. Users could highlight a section of the output and provide feedback specifically for that section. This can help identify issues at a more granular level.

```python
client.create_feedback(run_id, "contextual_feedback", score=1, comment="This part of the output was confusing.", context={"start": 10, "end": 50})
```

## 5. Automated Feedback

In addition to user feedback, automated feedback mechanisms can provide valuable insights. This could include metrics like perplexity, embedding distance, and string distance, as well as custom evaluators designed for specific tasks or domains.

```python
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

evaluation_config = RunEvalConfig(
    evaluators=[
        "embedding_distance",
        "string_distance",
        CustomEvaluator(),
    ]
)
run_on_dataset(
    client=client,
    dataset_name="<dataset_name>",
    llm_or_chain_factory=<LLM or constructor for chain or agent>,
    evaluation=evaluation_config,
    client=client,
)
```

# USED
# Rerun without context
# DOCS PAGES

## 6. User Engagement Metrics

In additiotn to the feedback on the model's output, it's also important to track user engagement metrics. This could include metrics like:

- Completion Rate: How often users complete the interaction without abandoning it.
- Return Rate: How often users return for additional interactions.
- Time Spent: How long users spend interacting with the model.

These metrics can provide insights into the overall user experience and highlight potential areas for improvement.

## 7. Feedback Analysis and Action

Collecting feedback is just the first step. It's crucial to regularly analyze the feedback, identify trends and common issues, and take action to improve the model and the user experience. This could involve training the model on new data, adjusting the model's parameters, or making changes to the user interface.

In conclusion, a comprehensive feedback mechanism should incorporate multiple dimensions of evaluation, allow for open-ended feedback, classify errors, provide contextual feedback, use automated feedback mechanisms, track user engagement metrics, and involve regular analysis and action. This will provide a more detailed and nuanced understanding of the model's performance and the user experience, leading to continuous improvement and a better product.


## 1. Accuracy Evaluation

We can use the QAEvalChain to grade the accuracy of a response against ground truth answers. This can provide a quantitative measure of how well the model's output matches the expected output.

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("qa")
evaluator.evaluate_strings(
    prediction="<model's output>",
    input="<user's input>",
    reference="<expected output>",
)
```

## 2. Model Comparison

If we have multiple models or versions of a model, we can use the PairwiseStringEvalChain or LabeledPairwiseStringEvalChain to compare their outputs. This can help us understand the differences between models and choose the best one.

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("pairwise_string")
evaluator.evaluate_strings(
    prediction1="<output of model 1>",
    prediction2="<output of model 2>",
)
```

## 3. Criteria Compliance

We can use the CriteriaEvalChain or LabeledCriteriaEvalChain to check whether an output complies with a set of criteria. This can help us ensure that the model's output meets certain standards or guidelines.

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("criteria")
evaluator.evaluate_strings(
    prediction="<model's output>",
    criteria="<list of criteria>",
)
```

## 4. Semantic Difference

We can use the EmbeddingDistanceEvalChain or PairwiseEmbeddingDistanceEvalChain to compute the semantic difference between a prediction and reference or between two predictions. This can help us understand how closely the model's output matches the expected output in terms of meaning.

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("embedding_distance")
evaluator.evaluate_strings(
    prediction="<model's output>",
    reference="<expected output>",
)
```

## 5. String Distance

We can use the StringDistanceEvalChain or PairwiseStringDistanceEvalChain to measure the string distance between a prediction and reference or between two predictions. This can provide a simple measure of how closely the model's output matches the expected output in terms of the exact string.

By incorporating these metrics into our feedback mechanism, we can provide a more detailed and nuanced evaluation of the model's performance. This can help us identify areas for improvement and make more informed decisions about how to improve the model.

```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("string_distance")
evaluator.evaluate_strings(
    prediction="<model's output>",
    reference="<expected output>",
)
```

Paraphrase Evaluation: Use a paraphrase generator to create multiple versions of the expected output and compare the model's output to these paraphrases. This can help evaluate the model's performance in tasks where there are multiple correct answers. You can use the ChatOpenAI model to create a chain for generating paraphrases as shown in the LangSmith documentation.

```python
from langchain import chat_models, schema, output_parsers, prompts
from langsmith import Client

paraphrase_llm = chat_models.ChatOpenAI(temperature=0.5)
prompt_template = prompts.ChatPromptTemplate.from_messages(
    [
        prompts.SystemMessagePromptTemplate.from_template(
            "You are a helpful paraphrasing assistant tasked with rephrasing text."
        ),
        prompts.SystemMessagePromptTemplate.from_template("Input: <INPUT>{query}</INPUT>"),
        prompts.HumanMessagePromptTemplate.from_template(
            "What are {n_paraphrases} different ways you could paraphrase the INPUT text?"
            " Do not significantly change the meaning."
            " Respond using numbered bullets. If you cannot think of any,"
            " just say 'I don't know.'"
        ),
    ]
)

paraphrase_chain = schema.LLMChain(
    llm=paraphrase_llm,
    prompt=prompt_template,
    output_parser=output_parsers.ListOutputParser(),
)

def evaluate_run(run: ls_schemas.Run) -> None:
    paraphrases = paraphrase_chain.invoke(
        {"query": run.outputs["output"][:3000], "n_paraphrases": 5},
    )
    for i, paraphrase in enumerate(paraphrases):
        evaluator = load_evaluator("embedding_distance")
        score = evaluator.evaluate_strings(
            prediction=run.outputs["output"],
            reference=paraphrase,
        )
        client.create_feedback(
            run.id,
            key=f"paraphrase_{i}_distance",
            score=score,
            feedback_source_type="MODEL",
        )
```

1. Interactive Feedback with Trajectory Evaluation: Utilize TrajectoryEvalChain to create an interactive feedback system where users can navigate through the conversation trajectory and provide feedback at each step. This can help identify specific points in the conversation where the model's performance deviated from the user's expectations. The TrajectoryOutputParser can be used to parse the output of this evaluation and provide more detailed feedback to the user.

2. Pairwise Comparison for A/B Testing: Implement PairwiseStringEvalChain or PairwiseEmbeddingDistanceEvalChain to perform A/B testing. You can generate two different outputs using different models or model configurations, and ask users to compare and provide feedback on which one they prefer and why.

3. Criteria-Based Feedback Game: Leverage CriteriaEvalChain to create a feedback game where users are asked to evaluate the model's output based on different criteria. This can make the feedback process more engaging for users and can also help collect more diverse feedback.

4. Embedding Distance for Semantic Similarity: Apply EmbeddingDistanceEvalChain to compute the semantic similarity between the model's output and the expected output. You can then use this information to provide feedback to the user about how closely the model's output matches the expected output in terms of meaning.

5. QA Evaluation for Interactive Learning: Use QAEvalChain to create an interactive learning system where the model asks the user questions and evaluates the user's answers. This can help the model learn from the user's feedback and improve its performance over time. You could also use QAGenerateChain to generate new question answering examples based on the feedback.

6. Regex Match for Pattern Recognition: Implement RegexMatchStringEvaluator to identify specific patterns in the model's output and provide feedback based on whether these patterns match the expected patterns.

7. String Distance for Textual Similarity: Utilize StringDistanceEvalChain to compute the textual similarity between the model's output and the expected output. You can then use this information to provide feedback to the user about how closely the model's output matches the expected output in terms of the exact string.

8. Agent Trajectory Evaluation for Multi-Turn Conversations: Apply AgentTrajectoryEvaluator to evaluate the model's performance in multi-turn conversations. This can help identify issues with the model's ability to maintain context and coherence over multiple turns.

9. Feedback Based on JSON Validity: If your model's output is supposed to be a valid JSON object, you can use JsonValidityEvaluator to check whether the output is valid JSON and provide feedback accordingly.

Remember, the goal of feedback is to improve the performance of your model. Therefore, it's important to continually iterate on your feedback mechanism based on what you learn from the feedback you collect.
