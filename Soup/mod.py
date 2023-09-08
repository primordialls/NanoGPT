fil = open("pilot.txt")
f = fil.read()

print(sorted(list(set(f))))

f = f.replace("\n\n","\n")

# # for line in f:
# #     if line.find("=") != -1:
# #         f.remove(line)

fil.close()
fil = open("pilot.txt","w")
fil.write(f)
#fil.write('\n'.join(f))