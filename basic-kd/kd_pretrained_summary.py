import os
from kd_utils import parse_pretrained_accuracies, get_student, get_teacher

pretrained_files = [x for x in os.listdir("./pretrained-basickd") if ".pt" in x]
pretrained_accuracies = parse_pretrained_accuracies(open("kd_pretrained_accuracies.txt"))

student_pretrained = [x for x in pretrained_files if "student" in x]
teacher_pretrained = [x for x in pretrained_files if "teacher" in x]
distilled_pretrained = [x for x in pretrained_files if "distilled" in x]

print("TYPE\t\t\t\tNUM_EPOCHS\tLR\t\tLAMBDA\t\tTEMP\t\tACCURACY")

for model in teacher_pretrained:
    x = model[:-3].split("-")
    acc = pretrained_accuracies[model]
    print(f"Teacher\t\t\t\t{x[1][2:]}\t\t{x[2][2:]}\t\tN.A.\t\tN.A.\t\t{acc}")

for model in student_pretrained:
    x = model[:-3].split("-")
    acc = pretrained_accuracies[model]
    print(f"Student (Baseline)\t\t{x[1][2:]}\t\t{x[2][2:]}\t\tN.A.\t\tN.A.\t\t{acc}")

for model in distilled_pretrained:
    x = model[:-3].split("-")
    acc = pretrained_accuracies[model]
    print(f"Student (Distilled)\t\t{x[1][2:]}\t\t{x[2][2:]}\t\t{x[3][5:]}\t\t{x[4][1:]}\t\t{acc}")

# print("TEACHER MODEL:")
# print(get_teacher())

# print("STUDENT MODEL:")
# print(get_student())
