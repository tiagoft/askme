from askme.makequestions.makequestion import HypothesisAboutCollection, QuestionMaker

collection = ["Example text 1", "Example text 2"]

def test_question_maker_default_config():
    qm = QuestionMaker()
    assert isinstance(qm, QuestionMaker)
    
def test_question_maker():
    qm = QuestionMaker()
    result = qm(collection)
    assert isinstance(result.output, HypothesisAboutCollection)
    assert isinstance(result.output.hypothesis, str)
    
