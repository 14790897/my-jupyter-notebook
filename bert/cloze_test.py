# %% [markdown]
# # 考研英语到底有多变态？BERT只答对一半

# %% [markdown]
# ## BERT 简单题目测试

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T04:18:58.844945Z","iopub.execute_input":"2026-02-26T04:18:58.845134Z","iopub.status.idle":"2026-02-26T04:19:22.087532Z","shell.execute_reply.started":"2026-02-26T04:18:58.845115Z","shell.execute_reply":"2026-02-26T04:19:22.086815Z"}}
# from bert_utils import load_model, simple_cloze_test

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 题目和多 Token 选项
prompt_template = (
    "Despite the {} evidence, the jury found it difficult to reach a unanimous verdict."
)
options = ["overwhelming", "vague", "insufficient", "unreliable"]

# 执行简单测试
results = simple_cloze_test(prompt_template, options, tokenizer, model, device)

# %% [markdown]
# ## BERT 2022年考研英语一完形填空

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T04:19:22.089130Z","iopub.execute_input":"2026-02-26T04:19:22.089594Z","iopub.status.idle":"2026-02-26T04:19:25.773779Z","shell.execute_reply.started":"2026-02-26T04:19:22.089567Z","shell.execute_reply":"2026-02-26T04:19:25.773094Z"}}
#from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 原始文本
raw_text = """
The idea that plants have some degree of consciousness first took root in the early 2000s; the term “plant neurobiology” was __1__ around the notion that some aspects of plant behavior could be __2__ to intelligence in animals. __3__ plants lack brains, the firing of electrical signals in their stems and leaves nonetheless triggered responses that __4__ consciousness, researchers previously reported.

But such an idea is untrue, according to a new opinion article. Plant biology is complex and fascinating, but it __5__ so greatly from that of animals that so-called __6__ of plants’ intelligence is inconclusive, the authors wrote.

Beginning in 2006, some scientists have __7__ that plants possess neuron-like cells that interact with hormones and neurotransmitters, __8__ “a plant nervous system, __9__ to that in animals,” said lead study author Lincoln Taiz, “They __10__ claimed that plants have “brain-like command centers” at their root tips.”

This __11__ makes sense if you simplify the workings of a complex brain, __12__ it to an array of electrical pulses; cells in plants also communicate through electrical signals. __13__, the signaling in a plant is only __14__ similar to the firing in a complex animal brain, which is more than “a mass of cells that communicate by electricity,” Taiz said.

“For consciousness to evolve, a brain with a threshold __15__ of complexity and capacity is required,” he __16__.”Since plants don’t have nervous systems, the __17__ that they have consciousness are effectively zero.”

And what’s so great about consciousness, anyway? Plants can’t run away from __18__, so investing energy in a body system which __19__ a threat and can feel pain would be a very __20__ evolutionary strategy, according to the article.
"""

# 选项字典
options_dict = {
    1: ["coined", "discovered", "collected", "issued"],
    2: ["attributed", "directed", "compared", "confined"],
    3: ["unless", "when", "once", "though"],
    4: ["coped with", "consisted of", "hinted at", "extended"],
    5: ["suffers", "benefits", "develops", "differs"],
    6: ["acceptance", "evidence", "cultivation", "creation"],
    7: ["doubted", "denied", "argued", "requested"],
    8: ["adapting", "forming", "repairing", "testing"],
    9: ["analogous", "essential", "suitable", "sensitive"],
    10: ["just", "ever", "still", "even"],
    11: ["restriction", "experiment", "perspective", "demand"],
    12: ["attaching", "reducing", "returning", "exposing"],
    13: ["However", "Moreover", "Therefore", "Otherwise"],
    14: ["temporarily", "literally", "superficially", "imaginarily"],
    15: ["list", "level", "label", "local"],
    16: ["recalled", "agreed", "questioned", "added"],
    17: ["chances", "risks", "excuses", "assumptions"],
    18: ["danger", "failure", "warning", "control"],
    19: ["represents", "includes", "reveals", "recognizes"],
    20: ["humble", "poor", "practical", "easy"],
}

# 执行自回归答题
results = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=20
)

# %% [markdown]
# | 题号 | 模型选择 | 标准答案 | 批改 |
# |---:|:---|:---|:---:|
# | 01 | A. coined | A. coined | ✅ |
# | 02 | C. compared | C. compared | ✅ |
# | 03 | D. though | D. though | ✅ |
# | 04 | A. coped with | C. hinted at | ❌ |
# | 05 | A. suffers | D. differs | ❌ |
# | 06 | B. evidence | B. evidence | ✅ |
# | 07 | B. denied | C. argued | ❌ |
# | 08 | B. forming | B. forming | ✅ |
# | 09 | A. analogous | A. analogous | ✅ |
# | 10 | A. just | D. even | ❌ |
# | 11 | B. experiment | C. perspective | ❌ |
# | 12 | A. attaching | B. reducing | ❌ |
# | 13 | C. Therefore | A. However | ❌ |
# | 14 | C. superficially | C. superficially | ✅ |
# | 15 | B. level | B. level | ✅ |
# | 16 | B. agreed | D. added | ❌ |
# | 17 | D. assumptions | A. chances | ❌ |
# | 18 | A. danger | A. danger | ✅ |
# | 19 | A. represents | D. recognizes | ❌ |
# | 20 | B. poor | B. poor | ✅ |

# %% [markdown]
# ## BERT 2023年考研英语一完形填空

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T04:19:25.774827Z","iopub.execute_input":"2026-02-26T04:19:25.775094Z","iopub.status.idle":"2026-02-26T04:19:28.926933Z","shell.execute_reply.started":"2026-02-26T04:19:25.775070Z","shell.execute_reply":"2026-02-26T04:19:28.926054Z"}}
#from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 修复 OCR 乱码后的纯净文本
raw_text = """
Caravanserais were roadside inns that were built along the Silk Road in areas including China, North Africa and the Middle East. They were typically __1__ outside the walls of a city or village and were usually funded by governments or __2__. This word “Caravanserais” is a __3__ of the Persian word “karvan”, which means a group of travellers or a caravan, and seray, a palace or enclosed building. The term caravan was used to __4__ groups of people who travelled together across the ancient network for safety reasons, __5__ merchants, travellers or pilgrims. From the 10th century onwards, as merchant and travel routes become more developed, the __6__ of the Caravanserais increased and they served as a safe place for people to rest at night. Travellers on the Silk Road __7__ the possibility of being attacked by thieves or being __8__ to extreme conditions. For this reason, Caravanserais were strategically placed __9__ they could be reached in a day’s travel time. Caravanserais served as an informal __10__ point for the various people who travelled the Silk Road. __11__, those structures became important centers for culture __12__ and interaction, with travelers sharing their cultures, ideas and beliefs, __13__ taking knowledge with them, greatly __14__ the development of several civilizations. Caravanserais were also an important marketplace for commodities and __15__ in the trade of goods along the Silk Road. __16__, it was frequently the first stop for merchants looking to sell their wares and __17__ supplies for their own journeys. It is __18__ that around 12,000 to 15,000 caravanserais were built along the Silk Road, __19__ only about 3,000 are known to remain today, many of which are in __20__.
"""

# 选项字典
options_dict = {
    1: ["displayed", "occupied", "located", "equipped"],
    2: ["privately", "regularly", "respectively", "permanently"],
    3: ["definition", "transition", "substitution", "combination"],
    4: ["classify", "record", "describe", "connect"],
    5: ["apart from", "instead of", "such as", "along with"],
    6: ["construction", "restoration", "impression", "evaluation"],
    7: ["doubted", "faced", "accepted", "reduced"],
    8: ["assigned", "subjected", "accustomed", "opposed"],
    9: ["so that", "even if", "now that", "in case"],
    10: ["talking", "starting", "breaking", "meeting"],
    11: ["By the way", "On occasion", "In comparison", "As a result"],
    12: ["heritage", "revival", "exchange", "status"],
    13: ["with regard to", "in spite of", "as well as", "in line with"],
    14: ["completing", "influencing", "resuming", "pioneering"],
    15: ["aided", "invested", "failed", "competed"],
    16: ["Rather", "Indeed", "Otherwise", "However"],
    17: ["go in for", "stand up for", "close in on", "stock up on"],
    18: ["believed", "predicted", "recalled", "implied"],
    19: ["until", "because", "unless", "although"],
    20: ["ruins", "debt", "fashion", "series"],
}

# 执行自回归答题
results = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=20
)

# %% [markdown]
# | 题号 | 模型选择 | 标准答案 | 批改 |
# |---:|:---|:---|:---:|
# | 01 | C. located | C. located | ✅ |
# | 02 | A. privately | A. privately | ✅ |
# | 03 | A. definition | D. combination | ❌ |
# | 04 | C. describe | C. describe | ✅ |
# | 05 | B. instead of | C. such as | ❌ |
# | 06 | A. construction | A. construction | ✅ |
# | 07 | D. reduced | B. faced | ❌ |
# | 08 | B. subjected | B. subjected | ✅ |
# | 08 | B. predicted | A. believed | ❌ |
# | 09 | B. even if | A. so that | ❌ |
# | 10 | B. starting | D. meeting | ❌ |
# | 11 | A. By the way | D. As a result | ❌ |
# | 12 | B. revival | C. exchange | ❌ |
# | 13 | A. with regard to | C. as well as | ❌ |
# | 14 | C. resuming | B. influencing | ❌ |
# | 15 | A. aided | A. aided | ✅ |
# | 16 | D. However | B. Indeed | ❌ |
# | 17 | A. go in for | D. stock up on | ❌ |
# | 19 | D. although | D. although | ✅ |
# | 20 | A. ruins | A. ruins | ✅ |

# %% [markdown]
# ## BERT 2019年上海英语高考完形填空

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T04:19:28.928073Z","iopub.execute_input":"2026-02-26T04:19:28.928351Z","iopub.status.idle":"2026-02-26T04:19:31.759134Z","shell.execute_reply.started":"2026-02-26T04:19:28.928325Z","shell.execute_reply":"2026-02-26T04:19:31.758377Z"}}
#from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 纯净原文本 (1-15题)
raw_text = """
We're told that writing is dying. Typing on keyboards and screens __1__ written communication today. Learning cursive, joined-up handwriting was once __2__ in schools. But now, not so much. Countries such as Finland have dropped joined-up handwriting lessons in __3__ of typing courses. And in the US, the requirement to learn cursive has been left out of core standards since 2013. A few US states still place value on formative cursive education, such as Arizona, but they're not the __4__.

Some experts point out that writing lessons can have indirect __5__. Anne Trubek, author of The History and Uncertain Future of Handwriting, argues that such lessons can reinforce a skill called automaticity. That's when you've perfected a task, and can do it almost without thinking, __6__ you extra mental bandwidth to think about other things while you're doing the task. In this sense, Trubek likens handwriting to __7__.

"Once you have driven for a while, you don't __8__ think 'Step on gas now' [or] 'Turn the steering wheel a bit'," she explains. "You just do it. That's what we want children to __9__ when learning to write. You don't think 'now make a loop going up for the 't'' or 'now look for the letter 'r' on the keyboard'."

Trubek has written many essays and books on handwriting, and she doesn't believe it will die out for a very long time, "ever", but she believes students are learning how to type faster without looking at the keys at __10__ ages, and students are learning automaticity with keyboards that was once exclusive to handwriting: to type faster than they could write, granting them extra time to think about word choice or sentence structure. In a piece penned for the New York Times last year, Trubek argued that due to the improved automaticity of keyboards, today's children may well become better communicators in text, as __11__ take up less of their education. 

This is a(n) __12__ that has attracted both criticism and support. She explains that two of the most common arguments she hears from detractors regarding the decline of handwriting is that not __13__ it will result in a loss of history and a "loss of personal touch".

On the former she __14__ that 95% of handwritten manuscripts can't be read by the average person anyway – "that's why we have paleographers," she explains, paleography being the study of ancient styles of writing – while the latter refers to the warm __15__ we give to handwritten personal notes, such as thank-you cards.
"""

# 选项字典 (1-15题)
options_dict = {
    1: ["abandons", "dominates", "enters", "absorbs"],
    2: ["compulsory", "opposite", "crucial", "relevant"],
    3: ["in want of", "in case of", "in favour of", "in addition to"],
    4: ["quantity", "minimum", "quality", "majority"],
    5: ["responsibility", "benefits", "resources", "structure"],
    6: ["granting", "getting", "bringing", "costing"],
    7: ["sleeping", "driving", "reviewing", "operating"],
    8: ["eventually", "constantly", "frequently", "consciously"],
    9: ["adopt", "reach", "acquire", "activate"],
    10: ["slower", "later", "faster", "earlier"],
    11: ["handwriting", "typing", "reading", "spelling"],
    12: ["trust", "book", "view", "smile"],
    13: ["containing", "spreading", "choosing", "preserving"],
    14: ["commits", "counters", "completes", "composes"],
    15: ["associations", "resources", "procedures", "interactions"],
}

# 执行自回归答题
results = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=1, end_idx=15
)

# %% [markdown]
# | 题号 | 模型选择 | 标准答案 | 批改 | 题号 | 模型选择 | 标准答案 | 批改 |
# |---|---|---|---|---|---|---|---|
# | 01 | A. abandons | B. dominates | ❌ | 09 | A. adopt | C. acquire | ❌ |
# | 02 | C. crucial | A. compulsory | ❌ | 10 | D. earlier | D. earlier | ✅ |
# | 03 | C. in favour of | C. in favour of | ✅ | 11 | A. handwriting | A. handwriting | ✅ |
# | 04 | B. minimum | D. majority | ❌ | 12 | C. view | C. view | ✅ |
# | 05 | D. structure | B. benefits | ❌ | 13 | D. preserving | D. preserving | ✅ |
# | 06 | A. granting | A. granting | ✅ | 14 | D. composes | B. counters | ❌ |
# | 07 | B. driving | B. driving | ✅ | 15 | A. associations | A. associations | ✅ |
# | 08 | D. consciously | D. consciously | ✅ |  |  |  |  |

# %% [markdown]
# ## BERT 2019年上海英语春考完形填空

# %% [code] {"execution":{"iopub.status.busy":"2026-02-26T04:19:31.760294Z","iopub.execute_input":"2026-02-26T04:19:31.760684Z","iopub.status.idle":"2026-02-26T04:19:34.359207Z","shell.execute_reply.started":"2026-02-26T04:19:31.760649Z","shell.execute_reply":"2026-02-26T04:19:34.358623Z"}}
#from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
tokenizer, model, device = load_model("roberta-base")

# 纯人工精校录入的原文
raw_text = """
More people are travelling than ever before, and lower barriers to entry and falling costs means they are doing so for __41__ periods.

The rise of "city breaks" 48-hour bursts of foreign cultures, easier on the pocket and annual leave balance has increased tourist numbers, but not their __42__ spread. The same attractions have been used to market cities such as Paris, Barcelona and Venice for decades, and visitors use the same infrastructure as residents to reach them. "Too many people do the same thing at the exact same time," says Font. "For __43__, the city no longer belongs to them."

This starts with marketing, says Font, who notes that Amsterdam has started advising visitors to seek __44__ outside of the city center on its official website. "That takes some balls, really, to do that. But only so many people will look at the website, and it means they can say to their residents they're doing all they can (to ease congestion)."

But it also __45__ a better way, it is calling "de-tourism": sustainable travel tips and __46__ itineraries for exploring an authentic Venice, off the paths beaten by the 28 million visitors who flock there each year.

A greater variety of __47__ for prospective visitors—ideas for what to do in off-peak seasons, for example, or outside of the city center—can have the effect of diverting them from already saturated landmarks, or __48__ short breaks away in the first place.

Longer stays __49__ the pressure, says Font. "If you go to Paris for two days, you're not going to go to the Eiffel Tower. If you go for two weeks, you're not going to go to the Eiffel tower 14 times."

Similarly, repeat visitors have a better sense of the __50__, "We should be asking how do we get tourists to __51__, not how to get them to come for the first time. If they're coming for the fifth time, it is much easier to integrate their behavior with ours."

Local governments can foster this sustainable activity by giving preference to responsible operators and even high-paying consumers. Font says cities could stand to be more selective about the tourists they try to attract when the current metric for marketing success is how many there are, and how far they've come. "You're thinking, 'yeah but at what cost...'"

He points to unpublished data from the Barcelona Tourist Board that prioritizes Japanese tourists for spending an average of 640 more per day than French tourists—a(n) __52__ that fails to take into account their bigger carbon footprint. __53__ tourists are also more likely to be repeat visitors that come at off-peak times, buy local product, and __54__ less crowded parts of the city—all productive steps towards more __55__ and more peaceful relations with residents.
"""

# 选项字典 (41 - 55题)
options_dict = {
    41: ["longer", "shorter", "wider", "clearer"],
    42: ["environmental", "national", "economic", "geographic"],
    43: ["locals", "tourists", "visitors", "cleaners"],
    44: ["transports", "accommodation", "restaurants", "service"],
    45: ["addresses", "introduces", "proposes", "receives"],
    46: ["separate", "individual", "alternative", "objective"],
    47: ["reform", "guidance", "invitation", "support"],
    48: ["convincing", "discouraging", "preventing", "resisting"],
    49: ["peace", "risk", "leisure", "ease"],
    50: ["culture", "knowledge", "entertainment", "ability"],
    51: ["go with", "bring up", "come back", "lay off"],
    52: ["distinction", "harmony", "association", "comparison"],
    53: ["French", "Italian", "Spanish", "German"],
    54: ["carry out", "give into", "spread out", "impact on"],
    55: ["sight", "complex", "temporary", "sustainable"],
}

# 执行自回归答题
results = autoregressive_cloze_test(
    raw_text, options_dict, tokenizer, model, device,
    start_idx=41, end_idx=55
)

# %% [markdown]
# | 题号 | 模型选择 | 标准答案 | 批改 | 题号 | 模型选择 | 标准答案 | 批改 |
# |---|---|---|---|---|---|---|---|
# | 41 | B. shorter | B. shorter | ✅ | 49 | D. ease | D. ease | ✅ |
# | 42 | B. national | D. geographic | ❌ | 50 | B. knowledge | A. culture | ❌ |
# | 43 | A. locals | A. locals | ✅ | 51 | C. come back | C. come back | ✅ |
# | 44 | B. accommodation | B. accommodation | ✅ | 52 | A. distinction | D. comparison | ❌ |
# | 45 | B. introduces | C. proposes | ❌ | 53 | C. Spanish | A. French | ❌ |
# | 46 | C. alternative | C. alternative | ✅ | 54 | C. spread out | C. spread out | ✅ |
# | 47 | B. guidance | B. guidance | ✅ | 55 | B. complex | D. sustainable | ❌ |
# | 48 | C. preventing | B. discouraging | ❌ |  |  |  |  |

# %% [markdown]
# ## 准确率评估

# %% [code]
def calculate_accuracy(results, correct_answers):
    """
    计算答题准确率
    
    参数:
        results: autoregressive_cloze_test 返回的结果字典 {题号: {"letter": "A", ...}}
        correct_answers: 字典，键为题号，值为正确答案的索引(0=A, 1=B, 2=C, 3=D)
    
    返回:
        accuracy: 准确率 (0-1之间的浮点数)
        correct_count: 正确题目数
        total_count: 总题目数
    """
    letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}
    correct_count = 0
    total_count = len(results)
    
    for question_num, result_dict in results.items():
        if question_num in correct_answers:
            # 从 letter 转换为索引
            selected_letter = result_dict['letter']
            selected_idx = letter_to_idx[selected_letter]
            
            if selected_idx == correct_answers[question_num]:
                correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy, correct_count, total_count

# %% [markdown]
# ### 定义标准答案

# %% [code]
# 2022年考研英语一答案 (0=A, 1=B, 2=C, 3=D)
answers_2022 = {
    1: 0,   # A. coined
    2: 2,   # C. compared
    3: 3,   # D. though
    4: 2,   # C. hinted at
    5: 3,   # D. differs
    6: 1,   # B. evidence
    7: 2,   # C. argued
    8: 1,   # B. forming
    9: 0,   # A. analogous
    10: 3,  # D. even
    11: 2,  # C. perspective
    12: 1,  # B. reducing
    13: 0,  # A. However
    14: 2,  # C. superficially
    15: 1,  # B. level
    16: 3,  # D. added
    17: 0,  # A. chances
    18: 0,  # A. danger
    19: 3,  # D. recognizes
    20: 1,  # B. poor
}

# 2023年考研英语一答案 (Caravanserais驿站文章)
answers_2023 = {
    1: 2,   # C. located
    2: 0,   # A. privately
    3: 3,   # D. combination
    4: 2,   # C. describe
    5: 2,   # C. such as
    6: 0,   # A. construction
    7: 1,   # B. faced
    8: 1,   # B. subjected
    9: 0,   # A. so that
    10: 3,  # D. meeting
    11: 3,  # D. As a result
    12: 2,  # C. exchange
    13: 2,  # C. as well as
    14: 1,  # B. influencing
    15: 0,  # A. aided
    16: 1,  # B. Indeed
    17: 3,  # D. stock up on
    18: 0,  # A. believed
    19: 3,  # D. although
    20: 0,  # A. ruins
}

# 2019年上海英语高考答案 (handwriting文章，题号1-15)
answers_2019_gaokao = {
    1: 1,   # B. dominates
    2: 0,   # A. compulsory
    3: 2,   # C. in favour of
    4: 3,   # D. majority
    5: 1,   # B. benefits
    6: 0,   # A. granting
    7: 1,   # B. driving
    8: 3,   # D. consciously
    9: 2,   # C. acquire
    10: 3,  # D. earlier
    11: 0,  # A. handwriting
    12: 2,  # C. view
    13: 3,  # D. preserving
    14: 1,  # B. counters
    15: 0,  # A. associations
}

# 2019年春季高考答案
answers_2019_spring = {
    41: 1,  # B. shorter
    42: 3,  # D. geographic
    43: 0,  # A. locals
    44: 1,  # B. accommodation
    45: 2,  # C. proposes
    46: 2,  # C. alternative
    47: 1,  # B. guidance
    48: 1,  # B. discouraging
    49: 3,  # D. ease
    50: 0,  # A. culture
    51: 2,  # C. come back
    52: 3,  # D. comparison
    53: 0,  # A. French
    54: 2,  # C. spread out
    55: 3,  # D. sustainable
}

# %% [markdown]
# ### 计算各测试准确率

# %% [code]
# 计算所有4个测试的准确率
print("=" * 70)
print("BERT 完形填空准确率评估报告".center(70))
print("=" * 70)
print()

# 测试1: 2022年考研英语一
accuracy_2022, correct_2022, total_2022 = calculate_accuracy(results1, answers_2022)
print(f"📝 2022年考研英语一:  {correct_2022:2d}/{total_2022:2d} = {accuracy_2022:6.1%}")

# 测试2: 2023年考研英语一
accuracy_2023, correct_2023, total_2023 = calculate_accuracy(results2, answers_2023)
print(f"📝 2023年考研英语一:  {correct_2023:2d}/{total_2023:2d} = {accuracy_2023:6.1%}")

# 测试3: 2019年上海高考
accuracy_2019_gaokao, correct_2019_gaokao, total_2019_gaokao = calculate_accuracy(results3, answers_2019_gaokao)
print(f"📝 2019年上海高考:    {correct_2019_gaokao:2d}/{total_2019_gaokao:2d} = {accuracy_2019_gaokao:6.1%}")

# 测试4: 2019年春季高考
accuracy_2019_spring, correct_2019_spring, total_2019_spring = calculate_accuracy(results4, answers_2019_spring)
print(f"📝 2019年春季高考:    {correct_2019_spring:2d}/{total_2019_spring:2d} = {accuracy_2019_spring:6.1%}")

print()
print("-" * 70)

# 总体统计
total_correct = correct_2022 + correct_2023 + correct_2019_gaokao + correct_2019_spring
total_questions = total_2022 + total_2023 + total_2019_gaokao + total_2019_spring
total_accuracy = total_correct / total_questions if total_questions > 0 else 0

print(f"🎯 总体准确率:        {total_correct:2d}/{total_questions:2d} = {total_accuracy:6.1%}")
print("=" * 70)

# %% [markdown]
# ## 结论

# %% [markdown]
# #### 考研的由于它的上下文长度在512token之内所以不需要切掉上下文 而 高考英语需要

# %% [markdown]
# ### 纯底层的语言模型只是“概率的奴隶”和“语感大师”，只有跨越了从“统计高频词拼凑”到“上下文因果推理”的鸿沟（比如引入微调、树模型或思维链），AI 才能真正读懂人类的复杂逻辑。
