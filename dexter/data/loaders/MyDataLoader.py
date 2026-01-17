from dexter.data.datastructures.evidence import Evidence
from dexter.data.loaders.BaseDataLoader import GenericDataLoader
from dexter.data.datastructures.question import Question
from dexter.data.datastructures.answer import Answer
from dexter.data.datastructures.sample import Sample
from dexter.config.constants import Split

# Modified MyDataLoader
class MyDataLoader(GenericDataLoader):

    def __init__(self, dataset: str, tokenizer="bert-base-uncased", config_path='test_config.ini', split=Split.TRAIN,
                 batch_size=None, corpus=None):
        super().__init__(dataset, tokenizer, config_path, split, batch_size)
        self.corpus = corpus

    def load_raw_dataset(self, split=Split.TRAIN):
        dataset = self.load_json(split)
        for data in dataset:
            # Create Question and Answer objects
            question = Question(data["question"], idx=data["_id"])
            answer = Answer(data["answer"], idx=data["_id"])

            # Ensure evidences are correctly processed
            evidences = []
            if "evidences" in data and isinstance(data["evidences"], list):
                for evidence in data["evidences"]:
                    if isinstance(evidence, dict) and "text" in evidence and "id" in evidence:
                        evidences.append(Evidence(text=evidence["text"], idx=evidence["id"]))

            # Append the Sample to raw_data
            self.raw_data.append(Sample(data["_id"], question, answer, evidences))

    def answers(self):
        answer_dict = {}
        for sample in self.raw_data:
            answer_dict[sample.idx] = sample.answer.text()
        return answer_dict

    def get_context_by_id(self, target_id):
        # Load the JSON data from the file
        data = self.load_json(Split.DEV)
        
        # Loop through the list of objects and find the one with the matching _id
        for obj in data:
            if obj.get('_id') == target_id:
                return obj['context']
        
        # Return None if the id is not found
        return None