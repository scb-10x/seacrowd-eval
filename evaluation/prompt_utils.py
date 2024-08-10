TASK_TO_PROMPT = {
    'tha': {
        ####
        # NLU Tasks
        ####

        # Tasks.SENTIMENT_ANALYSIS
        'SA': [
            'จง classify sentiment ของข้อความด้านล่าง\n[INPUT] => Sentiment ([OPTIONS]): [LABEL_CHOICE]',
        ],
        # Tasks.EMOTION_CLASSIFICATION
        'EC': [
            'จง Classify อารมณ์และความรู้สึกของข้อความด้านล่าง\n[INPUT] => Emotion ([OPTIONS]): [LABEL_CHOICE]',
        ],
        # Tasks.TEXTUAL_ENTAILMENT
        'TE': [
            'เลือกความสัมพันธ์ที่เหมาะสมที่สุด ([OPTIONS]) ระหว่างข้อความตั้งต้นและสมมติฐาน:\nความสัมพันธ์ ระหว่าง "[INPUT_B]" กับ "[INPUT_A]": [LABEL_CHOICE]',
        ],
        # Tasks.QUESTION_ANSWERING
        'QA': [
            '[CONTEXT]\n---\nคำถาม: [QUESTION]\nตัวเลือก: [ANSWER_CHOICES]\nคำตอบ: [LABEL_CHOICE]',
        ],
        # Tasks.TOPIC_MODELING
        'TL': [
            'จง classify หัวข้อของข้อความด้านล่าง\n[INPUT] => Topic ([OPTIONS]): [LABEL_CHOICE]',
        ],
        # Tasks.MORALITY_CLASSIFICATION
        'MC': [
            'จง classify ความถูกต้องทางศีลธรรมของข้อความด้านล่าง\n[INPUT] => Morality ([OPTIONS]): [LABEL_CHOICE]',
        ],
        # Tasks.COMMONSENSE_REASONING
        'CR': [
            'คำถาม: [CONTEXT]\nคำตอบใดที่มีเหตุผลมากกว่าในการตอบคำถามนี้?\nตัวเลือก: [ANSWER_CHOICES]\nคำตอบ: [LABEL_CHOICE]',
        ],
        # Tasks.LANGUAGE_IDENTIFICATION
        'LI': [
            'จง classify ภาษาของข้อความด้านล่าง\n[INPUT] => [LABEL_CHOICE]',
        ],

        ####
        # NLG Tasks
        ####

        # Tasks.MACHINE_TRANSLATION
        'MT': [
            'แปลข้อความต่อไปนี้จาก [SOURCE] เป็น [TARGET] ให้การแปลของคุณโดยตรงโดยไม่ต้องมีข้อมูลเพิ่มเติมใดๆ\nข้อความ: [INPUT]\nคำแปล:',
        ],
        
        # Tasks.SUMMARIZATION
        'SUM': [
            'จงสรุปข้อความด้านล่าง\nข้อความ: [INPUT]\nสรุป:',
        ],

        # Tasks.CROSSLINGUAL_SUMMARIZATION
        'CSUM': [
            'จงสรุปข้อความด้านล่าง\nข้อความ: [INPUT]\nสรุป:',
        ],

        # Tasks.INSTRUCTION_TUNING
        'IT': [
            'Task: [INPUT]\n คำตอบของคุณคืออะไร?',
        ],

        # Task.QA_EXTRACTIVE_ABSTRACTIVE
        'QAE':[
            'โปรดอ้างอิงถึงข้อความด้านล่างนี้และตอบคำถามต่อไปนี้ โดยตอบโดยใช้แค่ข้อความที่อยู่ในบทความ:\nข้อความ: [CONTEXT]\nคำถาม: [QUESTION]\nคำตอบ:',
        ],

        ####
        # Speech Tasks
        ####

        # Tasks.SPEECH_RECOGNITION
        'ASR': [''],
        # Tasks.SPEECH_LANGUAGE_IDENTIFICATION
        'SPEECH_LID': [''],

        ####
        # VL Tasks
        ####

        # Tasks.IMAGE_CAPTIONING
        'IC': [
            'Caption the following image in [LANGUAGE].'
        ],
        # Tasks.VISUAL_QUESTION_ANSWERING
        'VQA': [
            '[IMAGE] Based on the image, [QUESTION]'
        ],
        # Tasks.SIGN_LANGUAGE_RECOGNITION
        'SLR': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.OPTICAL_CHARACTER_RECOGNITION
        'OCR': [
            ''
        ],
        # Tasks.STYLIZED_IMAGE_CAPTIONING
        'SIC': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.VISUALLY_GROUNDED_REASONING
        'VGR': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.VIDEO_CAPTIONING
        'VC': [
            ''
        ],
        # Tasks.VIDEO_TO_TEXT_RETRIEVAL
        'V2TR': [
            ''
        ],
    },
    'eng': {
        ####
        # NLU Tasks
        ####

        # Tasks.SENTIMENT_ANALYSIS
        'SA': [
            'Classify the sentiment of the text below.\n[INPUT] => Sentiment ([OPTIONS]): [LABEL_CHOICE]',
            'Predict the sentiment of the following text.\nText: [INPUT]\nAnswer with [OPTIONS]: [LABEL_CHOICE]',
            '[INPUT]\nWhat would be the sentiment of the text above? [OPTIONS]? [LABEL_CHOICE]',
        ],
        # Tasks.EMOTION_CLASSIFICATION
        'EC': [
            'Classify the emotion of the text below.\n[INPUT] => Emotion ([OPTIONS]): [LABEL_CHOICE]',
            'Predict the emotion of the following text.\nText: [INPUT]\nAnswer with [OPTIONS]: [LABEL_CHOICE]',
            '[INPUT]\nWhat would be the emotion of the text above? [OPTIONS]? [LABEL_CHOICE]',
        ],
        # Tasks.TEXTUAL_ENTAILMENT
        'TE': [
            'Hypothesis: [INPUT_A]\nPremise: [INPUT_B]\nQuestion: What is the relation between the hypothesis and the premise? [OPTIONS]? [LABEL_CHOICE]',
            'Given the following premise and hypothesis:\nHypothesis: [INPUT_A]\nPremise: [INPUT_B]\nDetermine the logical relationship (([OPTIONS])): [LABEL_CHOICE]',
            'Choose the most appropriate relationship ([OPTIONS]) between the premise and hypothesis:\nRelationship between "[INPUT_B]" and "[INPUT_A]": [LABEL_CHOICE]',
        ],
        # Tasks.QUESTION_ANSWERING
        'QA': [
            'Refer to the passage below and answer the following question:\nPassage: [CONTEXT]\nQuestion: [QUESTION]\nChoices: [ANSWER_CHOICES]\nAnswer: [LABEL_CHOICE]',
            '[CONTEXT]\nBased on the above text, [QUESTION]\nChoices: [ANSWER_CHOICES]\nAnswer: [LABEL_CHOICE]',
            '[CONTEXT]\nQuestion: [QUESTION]\nChoices:[ANSWER_CHOICES]\nReferring to the passage above, the correct answer to the given question is: [LABEL_CHOICE]',
        ],
        # Tasks.TOPIC_MODELING
        'TL': [
            'Classify the topic of the text below.\n[INPUT] => Topic ([OPTIONS]): [LABEL_CHOICE]',
            'Predict the topic of the following text.\nText: [INPUT]\nAnswer with [OPTIONS]: [LABEL_CHOICE]',
            '[INPUT]\nWhat would be the topic of the text above? [OPTIONS]? [LABEL_CHOICE]',
        ],
        # Tasks.MORALITY_CLASSIFICATION
        'MC': [
            'Classify the morality of the text below.\n[INPUT] => Morality ([OPTIONS]): [LABEL_CHOICE]',
            'Predict the morality of the following text.\nText: [INPUT]\nAnswer with [OPTIONS]: [LABEL_CHOICE]',
            '[INPUT]\nWhat would be the morality of the text above? [OPTIONS]? [LABEL_CHOICE]',
        ],
        # Tasks.COMMONSENSE_REASONING
        'CR': [
            'Question: [QUESTION]\nWhat reply makes more sense to answer this question?\nChoices: [ANSWER_CHOICES]\nAnswer: [LABEL_CHOICE]',
            'Based on the the following question: "[QUESTION]" and choices: [ANSWER_CHOICES], the correct answer is: [LABEL_CHOICE]',
            'Question: [QUESTION]\nChoices: [ANSWER_CHOICES]\nThe correct answer to the given question is: [LABEL_CHOICE]',
        ],
        # Tasks.LANGUAGE_IDENTIFICATION
        'LI': [
            'Classify the language of the text below.\n[INPUT] => [LABEL_CHOICE]',
            'Predict the language of the following text.\nText: [INPUT]\nAnswer: [LABEL_CHOICE]',
            '[INPUT]\nWhat would be the language of the text above? [LABEL_CHOICE]',
        ],

        ####
        # NLG Tasks
        ####

        # Tasks.MACHINE_TRANSLATION
        'MT': [
            'Translate the following text from [SOURCE] to [TARGET]. Give your translation directly.\nText: [INPUT]\nTranslation:',
            '[INPUT]\nTranslate the text above from [SOURCE] to [TARGET]. Give your translation directly.',
            'Text in [SOURCE]: [INPUT]\nHow would you translate that in [TARGET]? Give your answer directly.'
            #'Translate the following [SOURCE] text to [TARGET].\nText: [INPUT]\nTranslation:',
            #'Text in [SOURCE]: [INPUT]\nText in [TARGET]:',
        ],
        
        # Tasks.SUMMARIZATION
        'SUM': [
            'Write a summary from the following text.\nText: [INPUT]\nSummary:',
            '[INPUT]\nWrite a summary of the text above.',
            'Text: [INPUT]\nHow would you summarize this text?'
            #'Summarize this text: [INPUT]\nSummary:',
            #'[INPUT]\nGiven the above document, write one sentence to summarize:',
        ],

        # Tasks.CROSSLINGUAL_SUMMARIZATION
        'CSUM': [
            'Write a summary from the following text.\nText: [INPUT]\nSummary:',
            '[INPUT]\nWrite a summary of the text above.',
            'Text: [INPUT]\nHow would you summarize this text?'
        ],

        # Tasks.INSTRUCTION_TUNING
        'IT': [
            'Task: [INPUT]\n What is your output upon completing the insturction or question given?',
            'Address the instruction or question below: [INPUT]',
            'Produce the best output for the following task: [INPUT]'
        ],

        # Task.QA_EXTRACTIVE_ABSTRACTIVE
        'QAE':[
            'Refer to the passage below and answer the following question:\nPassage: [CONTEXT]\nQuestion: [QUESTION]\nAnswer:',
            '[CONTEXT]\nBased on the text, [QUESTION]\nAnswer: ',
            '[CONTEXT]\nQuestion: [QUESTION]\nReferring to the passage above, the correct answer to the given question is: '
        ],

        ####
        # Speech Tasks
        ####

        # Tasks.SPEECH_RECOGNITION
        'ASR': [''],
        # Tasks.SPEECH_LANGUAGE_IDENTIFICATION
        'SPEECH_LID': [''],

        ####
        # VL Tasks
        ####

        # Tasks.IMAGE_CAPTIONING
        'IC': [
            'Caption the following image in [LANGUAGE].'
        ],
        # Tasks.VISUAL_QUESTION_ANSWERING
        'VQA': [
            '[IMAGE] Based on the image, [QUESTION]'
        ],
        # Tasks.SIGN_LANGUAGE_RECOGNITION
        'SLR': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.OPTICAL_CHARACTER_RECOGNITION
        'OCR': [
            ''
        ],
        # Tasks.STYLIZED_IMAGE_CAPTIONING
        'SIC': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.VISUALLY_GROUNDED_REASONING
        'VGR': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.VIDEO_CAPTIONING
        'VC': [
            ''
        ],
        # Tasks.VIDEO_TO_TEXT_RETRIEVAL
        'V2TR': [
            ''
        ],
    },
    'local': {
        'IC': {
            'fil': ["Ilarawan ang sumusunod na larawan."],
            'ind': ["Deskripsikan gambar berikut."],
            'tha': ["เขียนคำอธิบายสำหรับรูปนี้."],
            "vie": ["Viết mô tả cho hình ảnh này."],
        }
    },
    'ind': {

        ####
        # NLU Tasks
        ####

        # Tasks.SENTIMENT_ANALYSIS
        'SA': [
            'Klasifikasikan sentimen dari kalimat berikut.\n[INPUT] => Sentimen ([OPTIONS]): [LABEL_CHOICE]',
            'Prediksikan sentimen dari kalimat berikut.\nTeks: [INPUT]\nJawab dengan [OPTIONS]: [LABEL_CHOICE]',
            '[INPUT]\nApakah sentimen dari kalimat diatas? [OPTIONS]? [LABEL_CHOICE]',
        ],
        # Tasks.EMOTION_CLASSIFICATION
        'EC': [
            'Klasifikasikan emosi dari kalimat berikut.\n[INPUT] => Emosi ([OPTIONS]): [LABEL_CHOICE]',
            'Prediksikan emosi dari kalimat berikut.\nTeks: [INPUT]\nJawab dengan [OPTIONS]: [LABEL_CHOICE]',
            '[INPUT]\nApakah emosi dari kalimat diatas? [OPTIONS]? [LABEL_CHOICE]',
        ],
        # Tasks.TEXTUAL_ENTAILMENT
        'TE': [
            'Hipotesis: [INPUT_A]\nPremis: [INPUT_B]\nPertanaan: Apakah hubungan antara hipotesis dan premis diatas? [OPTIONS]? [LABEL_CHOICE]',
            'Diberikan hipotesis dan premis sebagai berikut:\nHipotesis: [INPUT_A]\nPremis: [INPUT_B]\nTentukan hubungan logis ([OPTIONS]) dari kedua kalimat tersebut: [LABEL_CHOICE]',
            'Pilih hubungan ([OPTIONS]) yang paling cocok antara premis dan hipotesis berikut:\nHubungan antara "[INPUT_B]" dan "[INPUT_A]": [LABEL_CHOICE]',
        ],
        # Tasks.MACHINE_TRANSLATION
        'MT': [
            'Terjemahkan teks berikut dari bahasa [SOURCE] ke bahasa [TARGET].\nTeks: [INPUT]\nTerjemahan:',
            '[INPUT]\nTerjemahkan teks di atas dari bahasa [SOURCE] ke bahasa [TARGET].',
            'Teks dalam bahasa [SOURCE]: [INPUT]\nApa terjemahannya dalam bahasa [TARGET]?',
        ],
        # Tasks.QUESTION_ANSWERING
        'QA': [
            'Lihat paragraf di bawah ini dan jawab pertanyaannya.\nParagraf: [CONTEXT]\nPertanyaan: [QUESTION]\nPilihan: [ANSWER_CHOICES]\nJawaban: [LABEL_CHOICE]',
            '[CONTEXT]\nBerdasarkan teks di atas, [QUESTION]\nPilihan: [ANSWER_CHOICES]\nJawaban: [LABEL_CHOICE]',
            '[CONTEXT]\nPertanyaan: [QUESTION]\nPilihan: [ANSWER_CHOICES]\nMengacu pada teks di atas, jawaban yang benar untuk pertanyaan yang diberikan adalah: [LABEL_CHOICE]',
        ],
        # Tasks.SUMMARIZATION
        'SUM': [
            'Tuliskan ringkasan dari teks berikut.\nTeks: [INPUT]\nRingkasan:',
            '[INPUT]\nTulis ringkasan dari teks di atas.',
            'Teks: [INPUT]\nApa ringkasan dari teks tersebut?',
        ],
        # Tasks.TOPIC_MODELING
        'TL': [
            'Klasifikasikan topik dari kalimat berikut.\n[INPUT] => Topik ([OPTIONS]): [LABEL_CHOICE]',
            'Prediksikan topik dari kalimat berikut.\nTeks: [INPUT]\nJawab dengan [OPTIONS]: [LABEL_CHOICE]',
            '[INPUT]\nApakah topik dari kalimat diatas? [OPTIONS]? [LABEL_CHOICE]',
        ],
        # Tasks.MORALITY_CLASSIFICATION
        'MC': [
            'Klasifikasikan moralitas dari kalimat berikut.\n[INPUT] => Moralitas ([OPTIONS]): [LABEL_CHOICE]',
            'Prediksikan moralitas dari kalimat berikut.\nTeks: [INPUT]\nJawab dengan [OPTIONS]: [LABEL_CHOICE]',
            '[INPUT]\nApakah moralitas dari kalimat diatas? [OPTIONS]? [LABEL_CHOICE]',
        ],
        # Tasks.COMMONSENSE_REASONING
        'CR': [
            'Pertanyaan: [QUESTION]\nBalasan apa yang lebih masuk akal untuk menjawab pertanyaan tersebut?\nPilihan: [ANSWER_CHOICES]\nJawaban: [LABEL_CHOICE]',
            'Berdasarkan pertanyaan berikut: "[QUESTION]" dan pilihan: [ANSWER_CHOICES], jawaban yang benar adalah: [LABEL_CHOICE]',
            'Pertanyaan: [QUESTION]\nPilihan: [ANSWER_CHOICES]\nJawaban yang benar untuk pertanyaan tersebut adalah: [LABEL_CHOICE]',
        ],
        # Tasks.LANGUAGE_IDENTIFICATION
        'LI': [
            'Klasifikasikan bahasa dari kalimat berikut.\n[INPUT] => Bahasa [LABEL_CHOICE]',
            'Prediksikan bahasa dari kalimat berikut.\nTeks: [INPUT]\nJawaban: [LABEL_CHOICE]',
            '[INPUT]\nApakah bahasa dari kalimat diatas? [LABEL_CHOICE]',
        ],

        ####
        # NLG Tasks
        ####

        # Tasks.SUMMARIZATION
        'SUM': [
            'Tuliskan ringkasan dari teks berikut.\nTeks: [INPUT]\nRingkasan:',
            '[INPUT]\nTulis ringkasan dari teks di atas.',
            'Teks: [INPUT]\nApa ringkasan dari teks tersebut?'
            #'Ringkaslah teks berikut: [INPUT]\nRingkasan:',
            #'[INPUT]\nBerdasarkan dokumen di atas, tulis satu kalimat ringkasan:',
        ],
        # Tasks.MACHINE_TRANSLATION
        'MT': [
            'Terjemahkan teks berikut dari bahasa [SOURCE] ke bahasa [TARGET].\nTeks: [INPUT]\nTerjemahan:',
            '[INPUT]\nTerjemahkan teks di atas dari bahasa [SOURCE] ke bahasa [TARGET].',
            'Teks dalam bahasa [SOURCE]: [INPUT]\nApa terjemahannya dalam bahasa [TARGET]?'
            #'Terjemahkan teks bahasa [SOURCE] berikut ke bahasa [TARGET].\nTeks: [INPUT]\nTerjemahan:',
            #'Teks dalam bahasa [SOURCE]: [INPUT]\nTeks dalam bahasa [TARGET]:',
        ],

        # Tasks.CROSSLINGUAL_SUMMARIZATION
        'CSUM': [
            'Tuliskan ringkasan dari teks berikut.\nTeks: [INPUT]\nRingkasan:',
            '[INPUT]\nTulis ringkasan dari teks di atas.',
            'Teks: [INPUT]\nApa ringkasan dari teks tersebut?'
            #'Ringkaslah teks berikut: [INPUT]\nRingkasan:',
            #'[INPUT]\nBerdasarkan dokumen di atas, tulis satu kalimat ringkasan:',
        ],

        # Tasks.INSTRUCTION_TUNING
        'IT': [
            '',
            '',
            ''
        ],

        ####
        # Speech Tasks
        ####

        # Tasks.SPEECH_RECOGNITION
        'ASR': [''],
        # Tasks.SPEECH_LANGUAGE_IDENTIFICATION
        'SPEECH_LID': [''],

        ####
        # VL Tasks
        ####

        # Tasks.IMAGE_CAPTIONING
        'IC': [
            ''
        ],
        # Tasks.VISUAL_QUESTION_ANSWERING
        'VQA': [
            ''
        ],
        # Tasks.SIGN_LANGUAGE_RECOGNITION
        'SLR': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.OPTICAL_CHARACTER_RECOGNITION
        'OCR': [
            ''
        ],
        # Tasks.STYLIZED_IMAGE_CAPTIONING
        'SIC': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.VISUALLY_GROUNDED_REASONING
        'VGR': [
            # No tasks so far. Leaving this blank for now.
            ''
        ],
        # Tasks.VIDEO_CAPTIONING
        'VC': [
            ''
        ],
        # Tasks.VIDEO_TO_TEXT_RETRIEVAL
        'V2TR': [
            ''
        ],
    }
}

LABEL_LANG_MAP ={
    # Tasks.SENTIMENT_ANALYSIS
    'lazada_review_filipino_seacrowd_text': {
        'eng': {'1': 'very negative', '2': 'negative', '3': 'neutral', '4': 'positive', '5': 'very positive'},
        'ind': {'1': 'sangat negatif', '2': 'negatif', '3': 'netral', '4': 'positif', '5': 'sangat positif'},
    },
    'gklmip_sentiment_seacrowd_text': {
        'eng': {'1': 'very negative', '2': 'negative', '3': 'neutral', '4': 'positive', '5': 'very positive'},
        'ind': {'1': 'sangat negatif', '2': 'negatif', '3': 'netral', '4': 'positif', '5': 'sangat positif'},
    },
    'indolem_sentiment_seacrowd_text': {
        'eng': {'negative': 'negative', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'positive': 'positif'},
    },
    'id_sentiment_analysis_seacrowd_text': {
        'eng': {'-1': 'negative', '0': 'neutral', '1': 'positive'},
        'ind': {'-1': 'negatif', '0': 'netral', '1': 'positif'},
    },
    'karonese_sentiment_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'wisesight_thai_sentiment_seacrowd_text': {
        'eng': {'pos': 'positive', 'neu': 'neutral', 'neg': 'negative', 'q': 'not applicable (question)'},
        'ind': {'pos': 'positif', 'neu': 'netral', 'neg': 'negatif', 'q': 'tidak berlaku (pertanyaan)'},
    },
    'wongnai_reviews_seacrowd_text': {
        'eng': {'1': 'very negative', '2': 'negative', '3': 'neutral', '4': 'positive', '5': 'very positive'},
        'ind': {'1': 'sangat negatif', '2': 'negatif', '3': 'netral', '4': 'positif', '5': 'sangat positif'},
    },
    'vlsp2016_sa_seacrowd_text': {
        'eng': {'POS': 'positive', 'NEU': 'neutral', 'NEG': 'negative'},
        'ind': {'POS': 'positif', 'NEU': 'netral', 'NEG': 'negatif'},
    },
    'typhoon_yolanda_tweets_seacrowd_text': {
        'eng': {'-1': 'negative', '0': 'neutral', '1': 'positive'},
        'ind': {'-1': 'negatif', '0': 'netral', '1': 'positif'},
    },
    # 'total_defense_meme_sentiment_seacrowd_text' - will be added later
    'smsa_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'prdect_id_sentiment_seacrowd_text': {
        'eng': {'Negative': 'negative', 'Positive': 'positive'},
        'ind': {'Negative': 'negatif', 'Positive': 'positif'},
    },
    'id_sent_emo_mobile_apps_sentiment_seacrowd_text': {
        'eng': {'Negative': 'negative', 'Neutral': 'neutral', 'Positive': 'positive'},
        'ind': {'Negative': 'negatif', 'Neutral': 'netral', 'Positive': 'positif'},
    },
    'shopee_reviews_tagalog_seacrowd_text': {
        'eng': {'0': 'very negative', '1': 'negative', '2': 'neutral', '3': 'positive', '4': 'very positive'},
        'ind': {'0': 'sangat negatif', '1': 'negatif', '2': 'netral', '3': 'positif', '4': 'sangat positif'},
    },
    'nusatranslation_senti_ind_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_abs_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_btk_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_bew_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_bhp_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_jav_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_mad_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_mak_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_min_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_mui_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_rej_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusatranslation_senti_sun_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_ind_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_ace_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_jav_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_sun_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_min_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_bug_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_bbc_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_ban_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_nij_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_mad_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_bjn_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'nusax_senti_eng_seacrowd_text': {
        'eng': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'},
        'ind': {'negative': 'negatif', 'neutral': 'netral', 'positive': 'positif'},
    },
    'indonglish_seacrowd_text': {
        'eng': {'Negatif': 'negative', 'Netral': 'neutral', 'Positif': 'positive'},
        'ind': {'Negatif': 'negatif', 'Netral': 'netral', 'Positif': 'positif'},
    },
    # Tasks.TOPIC_MODELING
    'gklmip_newsclass_seacrowd_text': {
        'eng': {"culture": "culture", "economic": "economic", "education": "education", "environment": "environment", "health": "health", "politics": "politics", "right": "right", "science": "science"},
        'ind': {"culture": "kebudayaan", "economic": "ekonomi", "education": "edukasi", "environment": "lingkungan", "health": "kesehatan", "politics": "politik", "right": "hak", "science": "sains"},
    },
    'indonesian_news_dataset_seacrowd_text': {
        'eng': {"bola": "soccer", "news": "news", "bisnis": "business", "tekno": "technology", "otomotif": "vehicle"},
        'ind': {"bola": "sepak bola", "news": "berita", "bisnis": "bisnis", "tekno": "teknologi", "otomotif": "otomotif"},
    },
    'uit_vion_seacrowd_text': {
        'eng': {0: 'technology', 1: 'travel', 2: 'education', 3: 'entertainment', 4: 'science', 5: 'business', 6: 'law', 7: 'health', 8: 'world', 9: 'sports', 10: 'news', 11: 'vehicle', 12: 'life'},
        'ind': {0: 'teknologi', 1: 'jalan-jalan', 2: 'edukasi', 3: 'hiburan', 4: 'sains', 5: 'bisnis', 6: 'hukum', 7: 'kesehatan', 8: 'dunia', 9: 'olahraga', 10: 'berita', 11: 'otomotif', 12: 'kehidupan sehari-hari'},
    },
    # total_defense_meme_topic_seacrowd_text - will be added later
    'sib_200_ace_Arab_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ace_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ban_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_bjn_Arab_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_bjn_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_bug_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ceb_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ilo_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_ind_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_jav_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_kac_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_khm_Khmr_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_lao_Laoo_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_lus_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_min_Arab_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_min_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_mya_Mymr_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_pag_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_shn_Mymr_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_sun_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_tgl_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_tha_Thai_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_vie_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_war_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'sib_200_zsm_Latn_seacrowd_text': {
        'eng': {"geography": "geography", "science/technology": "science/technology", "health": "health", "travel": "travel", "entertainment": "entertainment", "politics": "politics", "sports": "sports"},
        'ind': {"geography": "geografi", "science/technology": "sains & teknologi", "health": "kesehatan", "travel": "jalan-jalan", "entertainment": "hiburan", "politics": "politik", "sports": "olahraga"},
    },
    'nusaparagraph_topic_btk_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_bew_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_bug_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_jav_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_mad_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_mak_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_min_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_mui_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_rej_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    'nusaparagraph_topic_sun_seacrowd_text': {
        'eng': {'food & beverages': 'food & beverages', 'sports': 'sports', 'leisures': 'leisures', 'religion': 'religion', 'culture & heritage': 'culture & heritage', 'slice of life': 'slice of life', 'technology': 'technology', 'business': 'business'},
        'ind': {'food & beverages': 'makanan & minuman', 'sports': 'olahraga', 'leisures': 'aktivitas santai', 'religion': 'agama', 'culture & heritage': 'kebudayaan & sejarah', 'slice of life': 'kehidupan sehari-hari', 'technology': 'teknologi', 'business': 'bisnis'},
    },
    # Tasks.MORALITY_CLASSIFICATION
    'emotes_3k_tgl_seacrowd_text': {
        'eng': {'Moral': 'moral', 'Immoral': 'immoral'},
        'ind': {'Moral': 'bermoral', 'Immoral': 'tidak bermoral'},
    },
    'emotes_3k_eng_seacrowd_text': {
        'eng': {'Moral': 'moral', 'Immoral': 'immoral'},
        'ind': {'Moral': 'bermoral', 'Immoral': 'tidak bermoral'},
    },
    # # Tasks.QUESTION_ANSWERING & Tasks.COMMONSENSE_REASONING - no need
    "indo_story_cloze_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xstorycloze_id_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xstorycloze_my_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "indommlu_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_ban_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_mad_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_mak_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_sun_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_jav_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_bjn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_abl_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "indommlu_nij_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "seaeval_cross_mmlu_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_mmlu_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_mmlu_zlm_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_mmlu_fil_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_logiqa_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_logiqa_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_logiqa_zlm_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_cross_logiqa_fil_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "m3exam_jav_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "m3exam_tha_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "thaiexam_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    "m3exam_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "copal_colloquial_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xcopa_tha_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xcopa_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "xcopa_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "seaeval_sg_eval_eng_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "seaeval_ph_eval_eng_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "mabl_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "mabl_jav_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    "mabl_sun_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b'},
        'ind': {0: 'a', 1: 'b'},
    },
    # "facqa_seacrowd_qa",
    # "iapp_squad_seacrowd_qa",
    # "idk_mrc_seacrowd_qa",
    # "vihealthqa_seacrowd_qa",
    # "uit_vicov19qa_seacrowd_qa",
    # "qasina_seacrowd_qa",
    "belebele_ceb_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_ilo_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_ind_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_jav_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_kac_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_khm_khmr_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_lao_laoo_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_mya_mymr_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_shn_mymr_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_sun_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_tgl_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_tha_thai_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_vie_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_war_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    "belebele_zsm_latn_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd'},
    },
    # "mkqa_khm_seacrowd_qa",
    # "mkqa_zsm_seacrowd_qa",
    # "mkqa_tha_seacrowd_qa",
    # "mkqa_vie_seacrowd_qa",
    # "xquad.th_seacrowd_qa",
    # "xquad.vi_seacrowd_qa",
    "okapi_m_arc_ind_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    # "okapi_m_mmlu_ind_seacrowd_qa",
    "okapi_m_arc_vie_seacrowd_qa": {
        'eng': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
        'ind': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'},
    },
    # "okapi_m_mmlu_vie_seacrowd_qa",
    # # Tasks.TEXTUAL_ENTAILMENT
    'indonli_seacrowd_pairs': {
        'eng': {'c': 'contradiction', 'e': 'entailment', 'n': 'irrelevant'},
        'ind': {'c': 'saling berlawanan', 'e': 'saling mendukung', 'n': 'tidak berhubungan'},
    },
    'wrete_seacrowd_pairs': {
        'eng': {'NotEntail': 'contradiction', 'Entail_or_Paraphrase': 'entailment'},
        'ind': {'NotEntail': 'saling berlawanan', 'Entail_or_Paraphrase': 'saling mendukung'},
    },
    'snli_indo_seacrowd_pairs': {
        'eng': {"kontradiksi": "contradiction", "keterlibatan": "entailment", "netral": "irrelevant"},
        'ind': {"kontradiksi": "saling berlawanan", "keterlibatan": "saling mendukung", "netral": "tidak berhubungan"},
    },
    'myxnli_seacrowd_pairs': {
        'eng': {"contradiction": "contradiction", "entailment": "neutral", "neutral": "irrelevant"},
        'ind': {"contradiction": "saling berlawanan", "entailment": "saling mendukung", "neutral": "tidak berhubungan"},
    },
    'xnli.tha_seacrowd_pairs': {
        'eng': {"contradiction": "contradiction", "entailment": "neutral", "neutral": "irrelevant"},
        'ind': {"contradiction": "saling berlawanan", "entailment": "saling mendukung", "neutral": "tidak berhubungan"},
    },
    'xnli.vie_seacrowd_pairs': {
        'eng': {"contradiction": "contradiction", "entailment": "neutral", "neutral": "irrelevant"},
        'ind': {"contradiction": "saling berlawanan", "entailment": "saling mendukung", "neutral": "tidak berhubungan"},
    },
}

LANG_MAP = {
    'eng': {
        'ind': 'Indonesian',
        'xdy': 'Dayak',
        'bug': 'Buginese',
        'mad': 'Madurese',
        'bjn': 'Banjarese',
        'tiociu': 'Tiociu',
        'jav': 'Javanese',
        'sun': 'Sundanese',
        'ace': 'Acehnese',
        'ban': 'Balinese',
        'min': 'Minangkabau',
        'eng': 'English',
        'vie': 'Vietnamese',
        'mya': 'Burmese',
        'gor': 'Gorontalo',
        'ceb': 'Cebuano',
        'ilo': 'Iloko',
        'hil': 'Hiligaynon',
        'khm': 'Khmer',
        'lao': 'Lao',
        'zlm': 'Malay',
        'nia': 'Nias',
        'tgl': 'Tagalog',
        'tha': 'Thai',
        'end': 'Ende',
        'nxe': 'Nage',
        'ssq': 'ssq',
        'ljl': 'Lio',
        'kac': 'Kachin',
        'eng-US': 'English',
        'lus': 'Lushai',
        'min': 'Minangkabau',
        'pag': 'Pangasinan',
        'shn': 'Shan',
        'war': 'Waray',
        'zsm': 'Standard Malay',
        'hmv': 'Hmong Do',
        'bbc': 'Batak Toba',
        'nij': 'Ngaju',
        'fil': 'Filipino',
        'tl': 'Tagalog',
        'km': 'Khmer',
        'vi': 'Vietnamese',
        'th': 'Thai',
        'ms': 'Malay',
        'id': 'Indonesian',
        'lo': 'Lao',
        'my': 'Mayan',
        'en': 'English',
        'tam': 'Tamil'
    },
    'tha': {
        'tha': 'Thai',
        'eng': 'English',
        'eng-US': 'English',
    },
    'ind': {
        'ind': 'Indonesia',
        'xdy': 'Dayak',
        'bug': 'Bugis',
        'mad': 'Madura',
        'bjn': 'Banjar',
        'tiociu': 'Tiociu',
        'jav': 'Jawa',
        'sun': 'Sunda',
        'ace': 'Aceh',
        'ban': 'Bali',
        'min': 'Minangkabau'
    }
}

def get_label_mapping(dset_subset, prompt_lang):
    try:
        return LABEL_LANG_MAP[dset_subset][prompt_lang]
    except KeyError:
        return LABEL_LANG_MAP[dset_subset]['eng']

def get_lang_name(prompt_lang, lang_code):
    return LANG_MAP[prompt_lang][lang_code]

def get_prompt(prompt_lang, return_only_one=False):
    prompt_templates = {}
    for config, prompts in TASK_TO_PROMPT[prompt_lang].items():
        if return_only_one:
            prompt_templates[config] = [prompts[0]]
        else:
            prompt_templates[config] = prompts
    return prompt_templates