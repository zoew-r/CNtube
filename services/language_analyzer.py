"""
Language Analyzer module - Extract vocabulary and grammar points using LLM
"""
import os
from openai import OpenAI


class LanguageAnalyzer:
    """Analyze Chinese text for vocabulary and grammar using LLM."""
    
    def __init__(self):
        """Initialize the language analyzer with OpenAI client."""
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
    
    def analyze(self, transcription: str) -> dict:
        """
        Analyze the transcription for vocabulary and grammar points.
        
        Args:
            transcription: Chinese text to analyze
            
        Returns:
            Dictionary containing vocabulary and grammar analysis
        """
        if not self.client:
            # Return mock analysis when API key is not configured
            return self._mock_analysis(transcription)
        
        try:
            vocabulary = self._extract_vocabulary(transcription)
            grammar = self._extract_grammar_points(transcription)
            
            return {
                'vocabulary': vocabulary,
                'grammar_points': grammar
            }
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return self._mock_analysis(transcription)
    
    def _extract_vocabulary(self, text: str) -> list:
        """
        Extract key vocabulary from the text using LLM.
        
        Args:
            text: Chinese text to analyze
            
        Returns:
            List of vocabulary items with definitions and examples
        """
        prompt = f"""分析以下繁體中文文本，列出10-15個重要詞彙，適合中文學習者學習。

對於每個詞彙，請提供：
1. 詞彙 (繁體中文)
2. 拼音 (pinyin)
3. 英文翻譯
4. 詞性 (名詞、動詞、形容詞等)
5. 例句 (使用該詞彙的例句)

文本：
{text}

請以JSON格式回覆，格式如下：
[
  {{
    "word": "詞彙",
    "pinyin": "cí huì",
    "english": "vocabulary",
    "part_of_speech": "名詞",
    "example": "學習新詞彙很重要。"
  }}
]

只回覆JSON陣列，不要其他文字。"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一位專業的中文教師，專門幫助學生學習繁體中文。請根據HSK和TOCFL標準來選擇適當難度的詞彙。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up the response to extract JSON
            if content.startswith('```'):
                content = content.split('\n', 1)[1]
            if content.endswith('```'):
                content = content.rsplit('```', 1)[0]
            
            import json
            return json.loads(content)
        except Exception as e:
            print(f"Error extracting vocabulary: {e}")
            return []
    
    def _extract_grammar_points(self, text: str) -> list:
        """
        Extract grammar points from the text using LLM.
        
        Args:
            text: Chinese text to analyze
            
        Returns:
            List of grammar points with explanations and examples
        """
        prompt = f"""分析以下繁體中文文本，找出5-8個重要的語法點，適合中文學習者學習。

對於每個語法點，請提供：
1. 語法結構名稱
2. 詳細說明 (用英文解釋)
3. 結構公式 (如：Subject + 把 + Object + Verb)
4. 原文中的例子
5. 額外例句

文本：
{text}

請以JSON格式回覆，格式如下：
[
  {{
    "name": "把字句",
    "explanation": "The 把 (bǎ) construction is used to emphasize the result or effect of an action on an object.",
    "structure": "Subject + 把 + Object + Verb + Complement",
    "example_from_text": "我把書放在桌上。",
    "additional_examples": ["她把門關上了。", "請把這個給我。"]
  }}
]

只回覆JSON陣列，不要其他文字。"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一位專業的中文語法教師，專門教導外國學生學習中文語法。請參考現代漢語語法和對外漢語教學資源來解釋語法點。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            # Clean up the response to extract JSON
            if content.startswith('```'):
                content = content.split('\n', 1)[1]
            if content.endswith('```'):
                content = content.rsplit('```', 1)[0]
            
            import json
            return json.loads(content)
        except Exception as e:
            print(f"Error extracting grammar points: {e}")
            return []
    
    def _mock_analysis(self, text: str) -> dict:
        """
        Provide mock analysis when OpenAI API is not available.
        
        Args:
            text: Chinese text (not used in mock)
            
        Returns:
            Mock analysis dictionary
        """
        return {
            'vocabulary': [
                {
                    'word': '學習',
                    'pinyin': 'xué xí',
                    'english': 'to learn, to study',
                    'part_of_speech': '動詞',
                    'example': '我每天學習中文。'
                },
                {
                    'word': '影片',
                    'pinyin': 'yǐng piàn',
                    'english': 'video, film',
                    'part_of_speech': '名詞',
                    'example': '這部影片很有趣。'
                },
                {
                    'word': '語言',
                    'pinyin': 'yǔ yán',
                    'english': 'language',
                    'part_of_speech': '名詞',
                    'example': '中文是一種美麗的語言。'
                }
            ],
            'grammar_points': [
                {
                    'name': '是...的 Structure',
                    'explanation': 'The 是...的 construction is used to emphasize time, place, manner, or purpose of a past action.',
                    'structure': 'Subject + 是 + Time/Place/Manner + Verb + 的',
                    'example_from_text': '我是昨天來的。',
                    'additional_examples': ['他是在北京學中文的。', '這本書是我買的。']
                },
                {
                    'name': '了 Particle',
                    'explanation': 'The particle 了 indicates completed action or change of state.',
                    'structure': 'Verb + 了 (+ Object)',
                    'example_from_text': '我看了這個影片。',
                    'additional_examples': ['她吃了飯。', '天氣變冷了。']
                }
            ],
            'note': 'This is mock analysis. Configure OPENAI_API_KEY for real LLM analysis.'
        }
