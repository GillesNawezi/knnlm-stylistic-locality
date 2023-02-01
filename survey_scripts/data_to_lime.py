import pandas as pd
import pathlib


global_path = str(pathlib.Path(__file__).parent.resolve())


random_state =42
no_per_style = 5

styles = ["toxic","supportive","formal","informal","polite","impolite"]

file_name = global_path + "/input/survey_monkey.csv"
question_df = pd.read_csv(file_name)




questions = []

input_file_name = global_path + "/input/limesurvey_survey.txt"

#df = pd.read_table(file_name)
base_df = pd.read_csv(input_file_name, sep='\t', header=0, lineterminator='\n')



#Create Stndard Stylized Rubrik
stand_df = question_df.query(f"style=='No style'").copy()
stand_df =stand_df.sample(2, random_state=random_state).reset_index(drop=True)


group_col = {
    "id":int(0),
    "class":"G",
    "name": "No style",
    "relevance":int(1),
    "text":f"Comparing output of no_style model with stylised all style models output.",
    "language":"en" 
}
new_rows = []
new_rows.append(group_col)
for index, row in stand_df.iterrows():
    input_string = row['original']
    input_df = question_df.query(f'original=="{input_string}"').copy()

    for style in styles:
        style_df = input_df.query(f"style=='{style}'").copy()

        ix = style_df.index.values[0]
        #Create question
        question_id = str(ix)
        question_name = "Q"+ str(ix).zfill(2)

        question_dict = {
        "id":question_id, 
        "class": 'Q',
        "type/scale":"L",
        "name": question_name,
        "relevance":int(1),
        "text": f"Which text is more {style}?",
        "help": f"Please choose the answer that is more {style}.",
        "language":"en",  
        "mandatory":"Y",
        "other":"N"
        }


        answer_no_style = {
            "id":question_id,
            "class": 'A',
            "type/scale": int(0),
            "name": "AO01",
            "text": row["output"],
            "language":"en"
        }

        answer_all = {
            "id":question_id,
            "class": 'A',
            "type/scale": int(0),
            "name": "AO02",
            "text": style_df["output"].values[0],
            "language":"en"
        }

        new_rows.append(question_dict)
        new_rows.append(answer_no_style)
        new_rows.append(answer_all)


append_df = pd.DataFrame(new_rows)
output_df = pd.concat([base_df,append_df])





# Create Style Rubrik
count = int(1)
for style in styles:
    print(style)
    style_df = question_df.query(f"style=='{style}'").copy()
    style_df =style_df.sample(no_per_style, random_state=random_state).reset_index(drop=True)


    #Create Group Dict
    new_rows = []

    id = count

    group_col = {
        "id":id,
        "class":"G",
        "name": f"Full/Single {style}",
        "relevance":int(1),
        "text":f"Comparing output of {style} single style and all style models.",
        "language":"en" 
    }

    new_rows.append(group_col)

    for index, row in style_df.iterrows():

        #Create question
        question_id = int(index)
        question_name = "Q"+ str(index).zfill(2)

        question_dict = {
        "id":question_id, 
        "class": 'Q',
        "type/scale":"L",
        "name": question_name,
        "relevance":int(1),
        "text": f"Which text is more {style}?",
        "help": f"Please choose the answer that is more {style}.",
        "language":"en",  
        "mandatory":"Y",
        "other":"N"
        }


        answer_all = {
            "id":question_id,
            "class": 'A',
            "type/scale": int(0),
            "name": "AO01",
            "text": row["output"],
            "language":"en"
        }

        answer_single_style = {
            "id":question_id,
            "class": 'A',
            "type/scale": int(0),
            "name": "AO02",
            "text": row["output_single_style"],
            "language":"en"
        }

        new_rows.append(question_dict)
        new_rows.append(answer_all)
        new_rows.append(answer_single_style)

        append_df = pd.DataFrame(new_rows)


    output_df = pd.concat([output_df,append_df])
    count +=1
    print(count)

out_file_name = global_path + "/output/limesurvey_survey.tsv"

output_df.to_csv(out_file_name, sep='\t', index=False, float_format='%.0f')

