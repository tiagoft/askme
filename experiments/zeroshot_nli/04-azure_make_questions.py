from askme.makequestions import api, makequestion

def main():
    text_collection = [
        "The cat sat on the mat.",
        "The cat in in the box.",
        "The dog barked loudly.",
        "I like cats",
        "I like dogs",
        "The dog is in the yard.",
    ]
    model = api.build_model()
    question = makequestion.make_a_question_about_collection(
        collection=text_collection,
        model=model,
    )
    print("Generated Hypothesis:")
    print(question)
    print("Prompt tokens:", question.usage().input_tokens)
    print("Completion tokens:", question.usage().output_tokens)
    print("Total tokens:", question.usage().total_tokens)

    

if __name__ == "__main__":
    main()