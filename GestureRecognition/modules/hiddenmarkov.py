from SignalHub import GALY, bgr, get_nested_key, Module
from GestureRecognition.hmmclassifier import HMMClassifier

class HMMModule(Module):
    
    def __init__(self, outputSignal="markov", model_path="data/hmm.pkl", **kwargs):

        self.outputSignal = outputSignal
        self.model_path = model_path

        super().__init__(
            inputSignals=["config", "preprocessor"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="hiddenmarkov",
        )

    def start(self, data):
        try:
            self.classifier = HMMClassifier.load(self.model_path)
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")
            self.classifier = None
        return {}


    def step(self, data):
        trajectory = data.get("preprocessor")
        if trajectory is None or self.classifier is None:
            return {self.outputSignal: None}
        scores = self.classifier.decision_function(trajectory)
        best_label = max(scores, key=scores.get)
        result = {
            "label": best_label,
            "score": float(scores[best_label]),
            "all_scores": scores,
        }
        galy = GALY()
        galy.layer("markov")
        galy.putText(f"{best_label}  {result['score']:.1f}", (30, 50), color=(0, 255, 255))
        
        return {self.outputSignal: result, "galy": galy}

    def stop(self, data):
        pass