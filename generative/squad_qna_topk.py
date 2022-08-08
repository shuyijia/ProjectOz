import pandas as pd
from params import *
from datasets import Dataset
from squad_qna_pred import Squad_QnA_Prediction

def custom_predict(id_list, context_list, question_list, predictor):
    '''
    predictor: object of Squad_QnA_Prediction class
    '''
    assert len(id_list) == len(context_list) == len(question_list)

    data = {
        'id': id_list,
        'context': context_list,
        'question': question_list
    }

    df = pd.DataFrame.from_dict(data)
    ds = Dataset.from_pandas(df)

    validation_features = ds.map(
        predictor.prepare_validation_features,
        batched=True,
        remove_columns=ds.column_names
    )

    raw_predictions = predictor.trainer.predict(validation_features)

    validation_features.set_format(
        type=validation_features.format["type"], 
        columns=list(validation_features.features.keys())
    )

    final_predictions = predictor.postprocess_qa_predictions(
        ds,
        validation_features,
        raw_predictions.predictions
    )

    return final_predictions

if __name__ == '__main__':
    pred_head = Squad_QnA_Prediction()
    id_list = [1234]
    context_list = ['The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.']
    question_list = ['In what country is Normandy located?']

    out = custom_predict(id_list, context_list, question_list, pred_head)
    print(out)
