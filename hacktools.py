# Hack the results by rule based, bagging results or something else!
import csv

def clean_bert():
    tsv_file = open("./output/final/output.tsv", encoding='utf-8')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    print('yee')
    for row in read_tsv:
        if(row[4]=='time' and len(row[3])>4):
            continue
        if len(row[3])>5:
            continue
        print(row)
    tsv_file.close()

def try_read_next(reader):
    try:
        row = next(reader)
        return row
    except:
        return None

def clean():
    tsv_file = open("./to_merge/2.tsv", encoding='utf-8')
    out_file = open("./to_merge/2c.tsv", 'w',encoding='utf-8', newline='')
    reader2 = csv.reader(tsv_file, delimiter="\t")
    tsv_output = csv.writer(out_file, delimiter='\t')
    for row in reader2:
        try:
            row[2] = str(int(row[2]) - 1)
            tsv_output.writerow(row)
        except:
            tsv_output.writerow(row)
    tsv_file.close()
    out_file.close()

def ez_bagging(file1_path, file2_path): # file1 has high priority, put the one with better result here
    res = []
    tsv_file1 = open(file1_path, encoding='utf-8')
    tsv_file2 = open(file2_path, encoding='utf-8')
    reader1 = csv.reader(tsv_file1, delimiter="\t")
    reader2 = csv.reader(tsv_file2, delimiter="\t")
    row1 = try_read_next(reader1)
    row2 = try_read_next(reader2)
    cur_id=0
    cur_start=0
    cur_end=0
    while True:
        if not row1:
            while row2:
                res.append(row2)
                row2 = try_read_next(reader2)
            break
        if not row2:
            while row1:
                res.append(row1)
                row1 = try_read_next(reader1)
            break
        if row1==row2: # same
            res.append(row1)
            row1 = try_read_next(reader1)
            row2 = try_read_next(reader2)
        else: # indel & conflict
            if int(row1[0])<int(row2[0]) or (row1[0]==row2[0] and int(row1[1])<int(row2[1]) and int(row1[2])<int(row2[1])): # row1 insert
                res.append(row1)
                row1 = try_read_next(reader1)
            elif int(row1[0])>int(row2[0]) or (row1[0]==row2[0] and int(row1[1])>int(row2[1]) and int(row1[1])>int(row2[2])): # row2 insert
                # if row2[4] != 'med_exam':
                res.append(row2)
                row2 = try_read_next(reader2)
            else: #conflict, tag diff or s/e diff
                res.append(row1)
                # print(row1)
                # print(row2)
                row2 = try_read_next(reader2)
                row1 = try_read_next(reader1)
                
    tsv_file1.close()
    tsv_file2.close()

    with open('./to_merge/merged.tsv', 'w', encoding='utf-8', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for row in res:
            tsv_output.writerow(row)

def ez_bagging_ckip(file1_path, file2_path): # file1 has high priority, put the one with better result here
    res = []
    tsv_file1 = open(file1_path, encoding='utf-8')
    tsv_file2 = open(file2_path, encoding='utf-8')
    reader1 = csv.reader(tsv_file1, delimiter="\t")
    reader2 = csv.reader(tsv_file2, delimiter="\t")
    row1 = try_read_next(reader1)
    row2 = try_read_next(reader2)
    cur_id=0
    cur_start=0
    cur_end=0
    while True:
        if not row1:
            while row2:
                res.append(row2)
                row2 = try_read_next(reader2)
            break
        if not row2:
            while row1:
                res.append(row1)
                row1 = try_read_next(reader1)
            break
        if row1==row2: # same
            res.append(row1)
            row1 = try_read_next(reader1)
            row2 = try_read_next(reader2)
        else: # indel & conflict
            if int(row1[0])<int(row2[0]) or (row1[0]==row2[0] and int(row1[1])<int(row2[1]) and int(row1[2])<int(row2[1])): # row1 insert
                # print('insert r1 : ' + row1[0] +' '+ row2[0])
                res.append(row1)
                row1 = try_read_next(reader1)
            elif int(row1[0])>int(row2[0]) or (row1[0]==row2[0] and int(row1[1])>int(row2[1]) and int(row1[1])>int(row2[2])): # row2 insert
                # print('insert r2 : ' + row1[0] +' '+ row2[0])
                if row2[4] != 'med_exam' and row2[3][0]!='阿':
                    res.append(row2)
                row2 = try_read_next(reader2)
            else: #conflict, tag diff or s/e diff
                res.append(row1)
                row2 = try_read_next(reader2)
                row1 = try_read_next(reader1)
                
    tsv_file1.close()
    tsv_file2.close()

    with open('./to_merge/merged.tsv', 'w', encoding='utf-8', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for row in res:
            tsv_output.writerow(row)

def ez_bagging_sgns(file1_path, file2_path): # file1 has high priority, put the one with better result here
    res = []
    tsv_file1 = open(file1_path, encoding='utf-8')
    tsv_file2 = open(file2_path, encoding='utf-8')
    reader1 = csv.reader(tsv_file1, delimiter="\t")
    reader2 = csv.reader(tsv_file2, delimiter="\t")
    row1 = try_read_next(reader1)
    row2 = try_read_next(reader2)
    while True:
        if not row1:
            while row2:
                res.append(row2)
                row2 = try_read_next(reader2)
            break
        if not row2:
            while row1:
                res.append(row1)
                row1 = try_read_next(reader1)
            break
        if row1==row2: # same
            res.append(row1)
            row1 = try_read_next(reader1)
            row2 = try_read_next(reader2)
        else: # indel & conflict
            if int(row1[0])<int(row2[0]) or (row1[0]==row2[0] and int(row1[1])<int(row2[1]) and int(row1[2])<int(row2[1])): # row1 insert
                res.append(row1)
                row1 = try_read_next(reader1)
            elif int(row1[0])>int(row2[0]) or (row1[0]==row2[0] and int(row1[1])>int(row2[1]) and int(row1[1])>int(row2[2])): # row2 insert
                if row2[4] == 'profession':
                    res.append(row2)
                row2 = try_read_next(reader2)
            else: #conflict, tag diff or s/e diff
                res.append(row1)
                row2 = try_read_next(reader2)
                row1 = try_read_next(reader1)
    tsv_file1.close()
    tsv_file2.close()

    with open('./to_merge/merged_sgns.tsv', 'w', encoding='utf-8', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for row in res:
            tsv_output.writerow(row)

#clean()
ez_bagging_sgns('./to_merge/merged_ckip2.tsv', './to_merge/output.tsv')

# rule base
#普林	location
#徘徊 name
#芭乐	name