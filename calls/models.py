import datetime
import mongoengine as me

class TurnMetrics(me.EmbeddedDocument):
    turn_index          = me.IntField(default=0)
    user_input          = me.StringField()
    ai_response         = me.StringField()
    llm_latency_ms      = me.FloatField(default=0.0)
    tts_latency_ms      = me.FloatField(default=0.0)
    total_latency_ms    = me.FloatField(default=0.0)
    prompt_tokens       = me.IntField(default=0)
    completion_tokens   = me.IntField(default=0)
    total_tokens        = me.IntField(default=0)
    hallucination_score = me.FloatField(default=None)
    faithfulness_score  = me.FloatField(default=None)
    context_used        = me.ListField(me.StringField())
    evaluation_notes    = me.StringField(default="")
    timestamp           = me.DateTimeField(default=datetime.datetime.utcnow)

class CallSession(me.Document):
    meta = {"collection": "call_sessions", "ordering": ["-started_at"]}
    call_sid             = me.StringField(default="pending")
    to_number            = me.StringField(required=True)
    from_number          = me.StringField()
    status               = me.StringField(default="initiated")
    model_key            = me.StringField()
    model_id             = me.StringField()
    model_provider       = me.StringField()
    turns                = me.EmbeddedDocumentListField(TurnMetrics)
    full_transcript      = me.ListField(me.DictField())
    avg_llm_latency_ms   = me.FloatField(default=0.0)
    avg_total_latency_ms = me.FloatField(default=0.0)
    avg_hallucination    = me.FloatField(default=None)
    avg_faithfulness     = me.FloatField(default=None)
    total_turns          = me.IntField(default=0)
    call_duration_s      = me.FloatField(default=0.0)
    started_at           = me.DateTimeField(default=datetime.datetime.utcnow)
    ended_at             = me.DateTimeField()

    def finalize(self):
        self.ended_at    = datetime.datetime.utcnow()
        self.total_turns = len(self.turns)
        if self.turns:
            self.avg_llm_latency_ms   = sum(t.llm_latency_ms  for t in self.turns) / len(self.turns)
            self.avg_total_latency_ms = sum(t.total_latency_ms for t in self.turns) / len(self.turns)
            h = [t.hallucination_score for t in self.turns if t.hallucination_score is not None]
            f = [t.faithfulness_score  for t in self.turns if t.faithfulness_score  is not None]
            self.avg_hallucination = sum(h)/len(h) if h else None
            self.avg_faithfulness  = sum(f)/len(f) if f else None
        self.call_duration_s = max((self.ended_at - self.started_at).total_seconds(), 0)
        self.save()
