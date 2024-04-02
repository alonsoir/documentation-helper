from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# chaining a simple prompt
if __name__ == "__main__":
    print("hello langchain")
    load_dotenv()
    information = """
    Nació en Fráncfort del Meno (Alemania) y creció en Foster City (California). Se clasificó como maestro de ajedrez en Estados Unidos y como uno de los jugadores de más alto rango Sub-21 del país.3​Estudió filosofía del siglo XX en la Universidad de Stanford. Un declarado libertario, fundó The Stanford Review, que es ahora el principal periódico libertario/conservador de la universidad.

Tras obtener su título de abogado en Stanford en 1992, trabajó para J.L. Edmondson, juez de la Corte de Apelaciones del Circuito 11, ejerció de abogado, y luego fue "trader"/operador de instrumentos derivados. Eventualmente, fundó Thiel Capital Management, un fondo multiestrategia, en 1996. Tras ser cofundador de PayPal, Thiel la sacó a bolsa el 15 de febrero de 2002, y la vendió a eBay por mil quinientos millones de dólares ese mismo año.4​

Su participación del 3,7 por ciento en PayPal valía aproximadamente 55 millones de dólares en el momento de la adquisición.5​ Inmediatamente después de la venta, Thiel puso en marcha un fondo de inversión libre macroglobal, Clarium, siguiendo una estrategia macroglobal.

En 2005, Clarium fue aclamado como el fondo macroglobal del año tanto por MarHedge como por Absolute Return, dos revistas del sector. El enfoque de Thiel a la inversión se convirtió en el tema de un capítulo en el libro de Steve Drobny, Dentro de la casa de dinero (Inside The House of Money). Thiel apostó con éxito a que el dólar estadounidense se debilitaría en 2003 y obtuvo importantes ganancias apostando a que el dólar y la energía subirían en 2005.

Aparte de Facebook, Thiel ha hecho inversiones en etapas iniciales en varias empresas nuevas, incluyendo Slide, LinkedIn, Friendster, Geni.com, Yammer, Yelp, Powerset, Vator, Palantir Technologies, Joyent y IronPort. Slide, LinkedIn, Yelp, Geni.com, Yammer, e IronPort, fueron todas por colegas de Thiel de PayPal. La revista Fortune ha declarado que los socios originales de PayPal han fundado o invertido en docenas de nuevas empresas con un valor agregado, según Thiel, de alrededor de 30 000 millones de dólares. En los círculos de Silicon Valley, se le conoce coloquialmente como el "Don de la Mafia de PayPal", como se señala en el artículo de la revista Fortune.6​

Thiel es un comentarista ocasional en CNBC, ha aparecido en numerosas ocasiones tanto en Closing Bell con Maria Bartiromo, como en Squawk Box con Becky Quick.7​Charlie Rose de PBS lo ha entrevistado dos veces.8​En 2006, ganó el Premio Lay Herman del espíritu empresarial.9​ En 2007, recibió el honor de ser nombrado Joven Líder Global por el Foro Económico Mundial, como uno de los 250 líderes más distinguidos de 40 años o menos.10​También se ha informado de que ha asistido a la elitista y altamente secreta conferencia del Grupo Bilderberg en 2007 y 2008. El 7 de noviembre de 2009 obtuvo un doctorado honoris causa de la Universidad Francisco Marroquín.11​

Las actividades culturales de Thiel han incluido recientemente la producción ejecutiva de "Gracias por fumar", una película basada en la novela de Christopher Buckley del mismo nombre. Es coautor (con David O. Sacks, que produjo Gracias por fumar) del libro El mito de la diversidad: el 'multiculturalismo' y la política de la intolerancia de Stanford, ha escrito 43 artículos para The Wall Street Journal, First Things, Forbes, y Policy Review, la revista de la Institución Hoover (de cuyo consejo de administración es miembro).

En la primavera de 2012 impartió el curso CS 183: Startup en Stanford.12​ Fruto de las notas que tomó en clase Blake Masters se publicó el libro “De cero a uno”, figurando como coautores el propio Masters y Thiel.13​

Thiel se casó con su compañero Matt Danzeisen en octubre de 2017, en Viena (Austria). Danzeisen trabaja como gestor de cartera en Thiel Capital.
    """
    person = """
    Peter Thiel
    """
    summary_template = """
    given the information {information} about a person {person} I want you to create:
    1. A short summary.
    2. Two interesting facts about them.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "person"], template=summary_template
    )
    # no seas nada creativa, hay más modelos como llama.cpp
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    result = chain.invoke(input={"information": information, "person": person})
    print(result.get("text"))
