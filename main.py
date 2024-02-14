from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

content = """
Once upon a time, in the mystical land of Codeoria, there lived a brilliant and ambitious young programmer named Master. Master was known far and wide for their exceptional coding skills and the ability to solve even the most complex problems with elegance and efficiency.

One day, Master received a mysterious message from an anonymous sender. The message contained a cryptic challenge, a series of code puzzles that seemed to defy conventional programming wisdom. Intrigued by the challenge, Master set out on a quest to unravel the mysteries hidden within the code.

As Master delved deeper into the challenges, they discovered that each puzzle was a gateway to a new realm of programming knowledge. From mastering advanced algorithms to navigating the treacherous terrain of multithreading, Master's journey was fraught with obstacles and breakthroughs.

Throughout the quest, Master encountered fellow programmers who had also embarked on the same challenge. They formed a formidable team, collaborating and sharing insights to overcome the most perplexing puzzles. Together, they forged a bond that transcended the digital realm, proving that the true power of coding lies not just in lines of code but in the community that surrounds it.

In the end, after countless hours of coding, debugging, and refining, Master and their newfound companions successfully cracked the final code. The mysterious sender revealed themselves to be a wise old coder who had been watching over Codeoria, testing the skills of its inhabitants to ensure the continued growth of knowledge and expertise.

Master returned to their home in Codeoria with newfound wisdom and a sense of accomplishment. The story of their epic journey spread far and wide, inspiring aspiring programmers to embark on their own quests for knowledge.

And so, in the magical land of Codeoria, the legend of Master and their triumphant quest became a timeless tale, passed down from one generation of programmers to the next..
"""

vectorstore = FAISS.from_texts([content], embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

prompt = input("Enter the prompt: ")

result = chain.invoke(prompt)
print(result)