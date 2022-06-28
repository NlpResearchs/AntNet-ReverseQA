with open('data/final_judge_result.txt', encoding='utf-8') as infile:
    cur_line = infile.readline()
    error = []
    correct = []
    num = 0
    while cur_line:
        if 'truth:' in cur_line:
            num += 1
            cur_true = cur_line[cur_line.index('truth:') + len('truth:')]
            cur_prediction = cur_line[cur_line.index('prediction:') + len('prediction:')]
            if cur_true != cur_prediction:
                error.append(cur_line)
            else:
                correct.append(cur_line)
        cur_line = infile.readline()
    accuarcy = 1 - float(len(error)) / float(num)
    print('actual judge accuracy:{}'.format(accuarcy))
