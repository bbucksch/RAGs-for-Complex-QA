from typing import List


class Answer:
    def __init__(self, text: str, idx=None):
        self._text = text
        self._idx = idx

    def text(self)->str:
        return self._text

    def id(self):
        return self._idx

    def flatten(self)->List[str]:
        return [self._text]


class AmbigNQAnswer:
    def __init__(self, answers: List[List[List[Answer]]],idx=None):
        self.answers = answers
        self._idx = idx
    
    def id(self):
        return self._idx

    def flatten(self)->List[str]:
        flattened_answers = []
        for annotation in self.answers:
            for query in annotation:
                for answer in query:
                    flattened_answers.append(answer.text())
        return flattened_answers

class TATQAAnswer(Answer):
    def __init__(self, answers: List[Answer], idx=None):
        self.answers = answers
        self.idx = idx

    def text(self):
        return ",".join(self._text)

    def id(self):
        return self._idx
