from typing import List
import json
import random
import string
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

load_dotenv()


@tool
def escrever_json(filepath: str, data: dict) -> str:
    """Escreve um dicionáro JSON em um arquivo"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return f"Os dados JSON foram inseridos com sucesso no arquivo  '{filepath}' ({len(json.dumps(data))} caracteres)."
    except Exception as e:
        return f"Erro: {str(e)}"
    
@tool 
def ler_json(filepath: str) -> str:
    """Lê e retorna o conteúdo do arquivo JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except FileNotFoundError:
        return f"Erro: O arquivo '{filepath}' não foi encontrado!"
    except json.JSONDecodeError as e:
        return f"Erro: Arquivo JSON inválido - {str(e)}"
    except Exception as e:
        return f"Erro ao ler arquivo JSON - {str(e)}"
    
@tool
def gerar_exemplos(
    primeiros_nomes: List[str],
    ultimos_nomes: List[str],
    dominios: List[str]
) -> dict:
    
    """Gerador de registros em JSON"""

    if not primeiros_nomes:
        return {"Erro": "Primeiro nome não podem estar vazios!"}
    if not ultimos_nomes:
        return {"Erro": "Último nome não podem estar vazios!"}
    if not dominios:
        return {"Erro": "Domínio não podem estar vazios!"}
    
    usuarios = []
    count = len(primeiros_nomes)

    for i in range(count):
        primeiro = primeiros_nomes[i]
        ultimo = ultimos_nomes[i % len (ultimos_nomes)]
        dominio = dominios[i % len (dominios)]
        email = f"{primeiro.lower()}.{ultimo.lower()}@{dominio}"

        usuario = {
            "id": i + 1,
            "primeiroNome": primeiro,
            "ultimoNome": ultimo,
            "email": email,
            "registro": (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
        }
        usuarios.append(usuario)

    return {"usuarios": usuarios, "Total": len(usuarios)}

FERRAMENTAS = [escrever_json, ler_json, gerar_exemplos]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

MENSAGEM_PARA_AGENTE = (
    "Você é um gerador de dados, um assistente muito competente que gera dados de exemplo para aplicações. "
    "Para gerar usuários, você precisa de primeiros_nomes (list), ultimos_nomes (list) e dominios (list)."
    "Quando solicitado a salvar usuários, primeiro gere eles com a ferramenta, e imediatamente salve usando a função escrever_json"
)

agent = create_react_agent(llm, FERRAMENTAS, prompt=MENSAGEM_PARA_AGENTE)

def iniciar_agente(user_input: str, history: List[BaseMessage]) -> AIMessage:
    """Inicia o agente de IA"""
    try:
        result = agent.invoke(
            {"messages": history + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 50}
        )
        return result["messages"][-1]
    except Exception as e:
        return AIMessage(content=f"Erro: {str(e)}\n\nTente reformular a sua requisição ou seja mais específico")
    
if __name__ == "__main__":
     
    print("=" * 60)
    print("Agente 01 - Gerador de dados")
    print("=" * 60)
    print("Gere usuários e salve-os em um arquivo JSON.")
    print()
    print("Examples:")
    print("  - Gere usuários chamados Luiz e Pedro, ambos com o último nome Cardoso e salve-os no arquivo usuarios.json")
    print()
    print("Digite 'sair' para encerrar")
    print("=" * 60)

    history: List[BaseMessage] = []

    while True:
        user_input = input("Você: ").strip()

        # Check for exit commands
        if user_input.lower() in ['sair', ""]:
            print("Saindo!")
            break

        print("Agente: ", end="", flush=True)
        response = iniciar_agente(user_input, history)
        print(response.content)
        print()

        # Update conversation history
        history += [HumanMessage(content=user_input), response]
