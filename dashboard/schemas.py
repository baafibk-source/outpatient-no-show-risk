"""Request bodies. Input limits live here."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


Band = Literal["Low", "Medium", "High"]


class AppointmentIn(Strict):
    lead_time_days: int = Field(14, ge=0, le=365)
    age: int = Field(35, ge=0, le=110)
    weekday: Literal["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"] = "Tuesday"
    booked_hour: int = Field(10, ge=0, le=23)
    sms_received: bool = True
    conditions: List[Literal["Hypertension", "Diabetes", "Alcoholism"]] = Field([], max_length=3)
    prior_appts: int = Field(0, ge=0, le=500)
    prior_no_shows: int = Field(0, ge=0, le=500)
    days_since_last_appt: int = Field(0, ge=0, le=3650)
    multiple_appts_same_day: bool = False
    neighbourhood: Optional[str] = Field(None, max_length=100)


class ContextIn(Strict):
    """Which prediction an LLM endpoint is about. The server rebuilds the score, factors and action
    itself, from either a held-out appointment id or validated appointment inputs. Nothing the
    client says about the prediction (risk, factor text) is ever put in a prompt."""
    appointment_id: Optional[int] = Field(None, ge=0, le=10**12)
    appointment: Optional[AppointmentIn] = None

    @model_validator(mode="after")
    def exactly_one(self):
        if (self.appointment_id is None) == (self.appointment is None):
            raise ValueError("send exactly one of appointment_id or appointment")
        return self


class ReminderIn(Strict):
    context: ContextIn
    channel: Literal["sms", "call"] = "sms"


class ChatMessage(Strict):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatIn(Strict):
    context: ContextIn
    messages: List[ChatMessage] = Field(min_length=1, max_length=8)

    # Staff questions are capped at 500 characters; assistant replies may be longer
    @model_validator(mode="after")
    def check_question(self):
        if any(m.role == "user" and len(m.content) > 500 for m in self.messages) or self.messages[-1].role != "user":
            raise ValueError("questions must be at most 500 characters and the last message must be a question")
        return self
