from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub

# Konfiguracja - TUTAJ PODAJ SWOJE LICZBY
LICZBA_A = 15
LICZBA_B = 7
load_dotenv()

# Inicjalizacja LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)


@tool
def multiply_predefined(dummy_input: str = "") -> str:
    """Mnoży predefiniowane liczby z kodu (15 × 7)"""
    result = LICZBA_A * LICZBA_B
    return f"{LICZBA_A} × {LICZBA_B} = {result}"


@tool
def multiply_custom(numbers: str) -> str:
    """Mnoży dwie liczby podane w różnych formatach.
    Akceptuje formaty: '5*5', '5,5', '5 5', '5x5'.
    Przykłady użycia: multiply_custom('5*5') lub multiply_custom('12,8')"""

    if not numbers or numbers.strip() == "" or numbers == "None":
        return "Błąd: Nie podano liczb do pomnożenia"

    try:
        # Usuwamy białe znaki
        numbers = numbers.strip()

        # Obsługujemy różne separatory
        separators = ['*', '×', 'x', 'X', ',', ' ']
        parts = None

        for sep in separators:
            if sep in numbers:
                parts = [x.strip() for x in numbers.split(sep) if x.strip()]
                break

        if not parts or len(parts) != 2:
            return f"Błąd: Nie mogę rozpoznać dwóch liczb w '{numbers}'. Użyj formatu '5*5' lub '5,5' lub '5 5'"

        num1 = float(parts[0])
        num2 = float(parts[1])
        result = num1 * num2

        # Formatujemy wynik - jeśli to liczby całkowite, pokazujemy jako int
        if num1.is_integer():
            num1 = int(num1)
        if num2.is_integer():
            num2 = int(num2)
        if result.is_integer():
            result = int(result)

        return f"{num1} × {num2} = {result}"

    except ValueError:
        return f"Błąd: '{numbers}' zawiera nieprawidłowe liczby"
    except Exception as e:
        return f"Błąd: {str(e)}"


# Lista narzędzi
tools = [multiply_predefined, multiply_custom]

# Pobieramy prompt dla ReAct agenta
try:
    prompt = hub.pull("hwchase17/react")
except:
    # Fallback prompt jeśli hub nie działa
    from langchain.prompts import PromptTemplate

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

# Tworzymy agenta
agent = create_react_agent(llm, tools, prompt)

# Tworzymy executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Testy
if __name__ == "__main__":
    print("=== Test 1: Mnożenie 5*5 ===")
    try:
        result = agent_executor.invoke({"input": "Ile to jest 5*5"})
        print(f"Wynik: {result['output']}")
    except Exception as e:
        print(f"Błąd: {e}")

    print("\n=== Test 2: Predefiniowane liczby ===")
    try:
        result = agent_executor.invoke({"input": "Pomnóż predefiniowane liczby"})
        print(f"Wynik: {result['output']}")
    except Exception as e:
        print(f"Błąd: {e}")

    print("\n=== Test 3: Inne mnożenie ===")
    try:
        result = agent_executor.invoke({"input": "Ile to jest 12 razy 8"})
        print(f"Wynik: {result['output']}")
    except Exception as e:
        print(f"Błąd: {e}")
