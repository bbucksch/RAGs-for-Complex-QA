class BaseRetriver:
    def __init__(self) -> None:
        pass

    def encode_queries(self)->None:
        pass

    def encode_context(self)->None:
        pass

    def train(self)->None:
        pass

    def retrieve(self)->None:
        pass

class RetrieverFactory():
    def get_retreiver(self,retriever_name:str,config_path:str):
        if retriever_name.lower() == 'adore':
            from .dense.ADORE import ADORERetriever
            return ADORERetriever
        return None

