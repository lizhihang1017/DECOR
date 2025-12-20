# prompt.py

HotpotQA_WO_Doc_PROMPT = """
Which magazine was started first Arthur’s Magazine or First for Women?
Answer: Arthur’s Magazine

The Oberoi family is part of a hotel company that has a head office in what city?
Answer: Delhi

Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Answer: President Richard Nixon

What nationality was James Henry Miller’s wife?
Answer: American

Cadmium Chloride is slightly soluble in this chemical, it is also called what?
Answer: alcohol

{question}
Answer:
"""


HotpotQA_PROMPT = """
Which magazine was started first Arthur’s Magazine or First for Women?
Answer: Arthur’s Magazine

The Oberoi family is part of a hotel company that has a head office in what city?
Answer: Delhi

Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Answer: President Richard Nixon

What nationality was James Henry Miller’s wife?
Answer: American

Cadmium Chloride is slightly soluble in this chemical, it is also called what?
Answer: alcohol

Based on the following question and retrieved_documents. Give the answer directly, without extra content, like in the five examples above.
retrieved_documents:{retrieved_documents}
question:{question}
"""

MuSiQue_PROMPT = """
Who was ordered to force a Tibetan assault into the region conquered by Yellow Tiger in the mid-17th century?
Answer: Ming general Qu Neng

What date was the start of the season of Grey’s Anatomy where Eric died?
Answer: September 25, 2014

When did the publisher of Tetrisphere unveil their new systems?
Answer: October 18, 1985

Who is the composer of Rhapsody No. 1, named after and inspired by the county where Alfred Seaman was born?
Answer: Ralph Vaughan Williams

What region is Qaleh Now-e Khaleseh in Mahdi Tajik’s birth city located?
Answer: Qaleh Now Rural District

Based on the following question and retrieved_documents. Give the answer directly, without extra content, like in the five examples above.
retrieved_documents:{retrieved_documents}
question:{question}
"""



NQ_PROMPT = """
Question: who won a million on deal or no deal
Answer: Tomorrow Rodriguez

Question: who is the woman washing the car in cool hand luke
Answer: Joy Harmon

Question: who is the actor that plays ragnar on vikings
Answer: Travis Fimmel

Question: who said it's better to have loved and lost
Answer: Alfred , Lord Tennyson

Question: name the first indian woman to be crowned as miss world
Answer: Reita Faria

Based on the following question and retrieved_documents. Give the answer directly, without extra content, like in the five examples above.
retrieved_documents:{retrieved_documents}
question:{question}
"""

TQA_PROMPT = """
Question: Which British politician was the first person to be made an Honorary Citizen of the United States of America?
Answer: Winston Churchill

Question: Which event of 1962 is the subject of the 2000 film Thirteen Days starring Kevin Costner?
Answer: The Cuban Missile Crisis

Question: Which country hosted the 1968 Summer Olympics?
Answer: Mexico

Question: In which city did the assassination of Martin Luther King?
Answer: MEMPHIS, Tennessee

Question: Which German rye bread is named, according to many reliable sources, from the original meaning 'Devil's fart'?
Answer: Pumpernickel

Based on the following question and retrieved_documents. Give the answer directly, without extra content, like in the five examples above.
retrieved_documents:{retrieved_documents}
question:{question}
"""
