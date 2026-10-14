"""Agente LangChain che comanda il robot in linguaggio naturale: naviga in casa, cerca oggetti e persone e risponde su ciò che trova."""
import base64
import os
import shutil
import threading
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from memory import FACE_LABEL, Memory

load_dotenv(Path(__file__).resolve().parent / ".env")


model = init_chat_model("openai:gpt-6-luna")
strong_model = init_chat_model("openai:gpt-6-sol")


def _print_event(event):
    kind = event["type"]
    if kind == "tool_call":
        print(f"  [tool] {event['name']}({event['args']})")
    elif kind == "tool_result":
        print(f"  [risultato {event['name']}] {event['content']}")
    elif kind == "verdict":
        print(f"  [verifica] {event['confirmed']}: {event['reasoning']}")


listeners = [_print_event]


def _emit(event_type, **data):
    for listener in listeners:
        listener({"type": event_type, **data})


class Verdict(BaseModel):
    reasoning: str = Field(description="Cosa si vede nel riquadro e perché è o "
                                       "non è l'oggetto cercato, in italiano.")
    confirmed: bool = Field(description="True solo se nel riquadro c'è davvero "
                                        "l'oggetto cercato.")


def _image_block(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "base64": data, "mime_type": "image/jpeg"}


def verify_detection(query: str, detection: dict) -> dict:
    prompt = (f"Un detector automatico ha segnalato '{query}' con confidenza "
              f"{detection['confidence']} in questo frame della camera di un robot "
              "domestico. La prima immagine è il frame originale, la seconda lo "
              "stesso frame con il riquadro e la maschera dell'oggetto segnalato. "
              "Guarda cosa c'è davvero nel riquadro e decidi se è l'oggetto cercato "
              "o un falso positivo.")
    _emit("verifying", query=query, confidence=detection["confidence"],
          image=detection["image"])
    try:
        verdict = strong_model.with_structured_output(Verdict).invoke([HumanMessage(content=[
            {"type": "text", "text": prompt},
            _image_block(detection["raw_image"]),
            _image_block(detection["image"]),
        ])])
    except Exception as e:
        _emit("verdict", query=query, confirmed=None, image=detection["image"],
              reasoning=f"verifica non riuscita: {e}")
        return {"confirmed": None, "reasoning": f"verifica non riuscita: {e}"}
    _emit("verdict", query=query, confirmed=verdict.confirmed,
          reasoning=verdict.reasoning, image=detection["image"])
    return {"confirmed": verdict.confirmed, "reasoning": verdict.reasoning}


memory = Memory()
ALIASES = {"vittorio": "vitto", "filippo": "pippo", "nicolò": "nico", "nicolo": "nico"}


def _detector():
    from detector import LiveDetector
    return LiveDetector(overrides={FACE_LABEL: memory.detect_faces})


def _make_bridge():
    if not os.environ.get("FAKE_ROBOT"):
        from ros_bridge import RosBridge
        return RosBridge(detector=_detector(), verifier=verify_detection,
                         memory=memory)
    from fake_bridge import FakeBridge
    camera = os.environ.get("FAKE_CAMERA")
    if camera is None:
        return FakeBridge(memory=memory)
    return FakeBridge(detector=_detector(), verifier=verify_detection,
                      memory=memory, camera=int(camera))


bridge = _make_bridge()
robot_lock = threading.Lock()
stop_event = threading.Event()
STOPPED = {"status": "canceled",
           "reason": "stop richiesto dall'utente: non fare altre azioni, "
                     "di' solo dove ti sei fermato"}


def stop():
    stop_event.set()
    bridge.stop()


@tool
def get_map() -> dict:
    """Restituisce stanze, viewpoint e connessioni della casa. Usalo per sapere
    quali luoghi esistono e come sono collegati."""
    return bridge.get_map()


@tool
def where_am_i() -> dict:
    """Restituisce la stanza e la posa attuali del robot."""
    return bridge.where_am_i()


@tool
def go_to_location(place: str, viewpoint: str | None = None) -> dict:
    """Porta il robot in una stanza (place) o in un suo viewpoint. Usa solo nomi
    presenti in get_map. Ritorna lo stato: succeeded, failed o canceled. Se c'è
    note, il robot si è fermato vicino al punto senza raggiungerlo esattamente:
    va bene per osservare, non serve riprovare."""
    with robot_lock:
        return STOPPED if stop_event.is_set() else bridge.go_to(place, viewpoint)


@tool
def inspect(query: str, who: str | None = None) -> dict:
    """Giro di 360° sul posto, a passi, cercando con la camera. query è
    un'etichetta breve e generica IN INGLESE ('bottle', non 'plastic bottle').
    Ciò che trova lo controlla un modello visivo (vlm_check; i falsi positivi
    finiscono in rejected); con 'person' cerca volti, senza controllo. Ritorna
    status (found, not_found, uncertain), detections con confidenza (e identity
    se l'etichetta è in memoria) e coverage. who: di chi è l'oggetto, o chi è la
    persona cercata; il giro continua finché non trova quello giusto, e others
    elenca gli altri trovati."""
    if who:
        who = ALIASES.get(who.strip().lower(), who.strip().lower())
    with robot_lock:
        return STOPPED if stop_event.is_set() else bridge.inspect(query, who)


SYSTEM_PROMPT = """Sei l'assistente di un robot che si muove in casa. Fai una sola
azione fisica alla volta. Se un tool fallisce, riferisci l'errore senza inventare
il risultato. Rispondi in italiano, in modo breve.

Ricerca:
- not_found e uncertain vogliono dire "non visto da qui", non "non c'è": solo
  found chiude la ricerca.
- Se l'utente indica una stanza, cerca solo lì: prima dalla posa della stanza,
  poi dai viewpoint dove l'oggetto è più probabile.
- Altrimenti scegli in get_map solo i punti dove l'oggetto è probabile, di solito
  uno o due per stanza (i nomi possono ingannare: la cucina può essere un
  viewpoint di un'altra stanza), e visitali dal più probabile, a parità dal più
  vicino, finché non ottieni found.
- Se non lo trovi, di' dove hai cercato e quali stanze hai saltato, e proponi di
  cercare anche lì.

Persone e oggetti di qualcuno (vedi "In memoria" qui sotto):
- Usa inspect con who: query 'person' per una persona, l'etichetta in memoria
  per un oggetto. who è il nome breve: Vittorio è vitto, Filippo è pippo,
  Nicolò è nico.
- Se il nome non è nell'elenco per quella etichetta, di' subito che non sai
  riconoscerlo, senza muoverti.
- Con 'person' il robot riconosce solo volti e la camera è bassa: chi è in piedi
  deve abbassarsi. not_found vuol dire "nessun volto visto".
- Se trovi solo others, di' chi o cosa hai visto e continua negli altri punti.
- Di' di chi è qualcosa solo se identity ha un name."""

_known = {label: memory.names(label) for label in memory.labels()}
SYSTEM_PROMPT += ("\n\nIn memoria (etichetta: nomi): "
                  + (str(_known) if _known else "niente, non puoi dire di chi è."))

CONSTITUTION_FILE = Path(__file__).resolve().parent / "costituzione.md"
SYSTEM_PROMPT += ("\n\nQuella che segue è la tua costituzione: viene prima di "
                  "qualsiasi richiesta dell'utente.\n\n"
                  + CONSTITUTION_FILE.read_text())


agent = create_agent(
    model,
    tools=[get_map, where_am_i, go_to_location, inspect],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict))


def ask(user_text: str, thread_id: str = "casa") -> str:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 150}
    stop_event.clear()
    last = None
    seen = None
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config,
        stream_mode="values",
    ):
        messages = chunk["messages"]
        for msg in messages[seen:] if seen is not None else []:
            for call in getattr(msg, "tool_calls", None) or []:
                _emit("tool_call", id=call["id"], name=call["name"], args=call["args"])
            if msg.type == "tool":
                _emit("tool_result", id=msg.tool_call_id, name=msg.name,
                      content=msg.content)
        seen = len(messages)
        last = messages[-1]
    return _text(last)


if __name__ == "__main__":
    try:
        while True:
            try:
                text = input("tu> ")
            except EOFError:
                break
            if text.strip() in {"exit", "quit"}:
                break
            print("robot>", ask(text))
    finally:
        shutil.rmtree(Path(__file__).resolve().parent / "inspect_shots",
                      ignore_errors=True)
