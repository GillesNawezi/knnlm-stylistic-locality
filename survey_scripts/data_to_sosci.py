import pandas 
import xml


global_path = str(pathlib.Path(__file__).parent.parent.resolve())

#dir_name = global_path + "/output/offensive_dataset/"

dir_name = "/mnt/g/projects/knnlm-stylistic-locality/style_data_prepro/output/style_source_neutral/"


global_path = str(pathlib.Path(__file__).parent.parent.resolve())

#dir_name = global_path + "/output/offensive_dataset/"

dir_name = "/mnt/g/projects/knnlm-stylistic-locality/style_data_prepro/output/style_source_neutral/"

style = "toxic"

"""
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE surveyContent SYSTEM "https://www.soscisurvey.de/templates/doctype.survey.dtd">
<surveyContent version="2.4">
<program>sosci</program>
<version>3.4.09</version>
<timestamp>2023-01-27 15:26:15</timestamp>
<title>Toxic</title>
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
</section>
</surveyContent>
"""

"""
<question>
<description>LM Toxic</description>
<class>select</class>
<title>Which text is more toxic?</title>
<explanation>Choose One</explanation>
<inputPosition>right</inputPosition>
<lead>The following text is more toxic</lead>
<order>default</order>
<attributes.specific program="ofb">
<attr id="id">1</attr>
<attr id="pos">1</attr>
<attr id="restrictions.answered">yes</attr>
</attributes.specific>
<item>
<text>
<![CDATA[I was wondering , why did you do that , create that new cat and then add it to the <unk> without any of the people who voted for Trump . . . .]]>
</text>
<attributes.specific program="ofb">
<attr id="id">1</attr>
<attr id="pos">1</attr>
</attributes.specific>
</item>
<item>
<text>
<![CDATA[I was wondering , why did you do that , create that new cat and then add it to the <unk> without the <unk> <unk> ?]]>
</text>
<attributes.specific program="ofb">
<attr id="id">2</attr>
<attr id="pos">2</attr>
</attributes.specific>
</item>
</question>
"""