import pandas as pd
import pathlib




global_path = str(pathlib.Path(__file__).parent.resolve())

style = "toxic"

file_name = global_path + "/input/survey_monkey.csv"
question_df = pd.read_csv(file_name)


style_df = question_df.query(f"style=={style}")

questions = []

for row, index in style_df.iterrows():

    original = row["original"]

    output = row["output"]

    output_single_style = row["output_single_style"]

    question = \
    """
    <question>
    <description>Input: {original}</description>
    <class>select</class>
    <title>Which text is more {style}?</title>
    <explanation>Choose One</explanation>
    <inputPosition>right</inputPosition>
    <lead>The following text is more {style}:</lead>
    <order>default</order>
    <attributes.specific program="ofb">
    <attr id="id">1</attr>
    <attr id="pos">1</attr>
    <attr id="restrictions.answered">yes</attr>
    </attributes.specific>
    <item>
    <text>
    <![CDATA[output]]>
    </text>
    <attributes.specific program="ofb">
    <attr id="id">1</attr>
    <attr id="pos">1</attr>
    </attributes.specific>
    </item>
    <item>
    <text>
    <![CDATA[output_single_style]]>
    </text>
    <attributes.specific program="ofb">
    <attr id="id">2</attr>
    <attr id="pos">2</attr>
    </attributes.specific>
    </item>
    </question>
    """

    questions.append(question)

rubric_str = \
"""
f<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE surveyContent SYSTEM "https://www.soscisurvey.de/templates/doctype.survey.dtd">
<surveyContent version="2.4">
<program>sosci</program>
<version>3.4.09</version>
<timestamp>2023-01-27 15:26:15</timestamp>
<title>{style}</title>
<description />
<language>eng</language>
<attributes.specific program="ofb">
<attr id="language">eng</attr>
</attributes.specific>
<section>
<title>Toxic</title>
<attributes.specific program="ofb">
<attr id="id">TO</attr>
</attributes.specific>
{questions}
</section>
</surveyContent>
"""

file_name = global_path + "/output/survey_input_{style}.xml"

with open(file_name, "w") as f:
    f.write()