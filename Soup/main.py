from bs4 import BeautifulSoup
import requests

def grabScript(url):
    r = requests.get(url, auth=('user', 'pass'))

    soup = BeautifulSoup(r.text,"html.parser")
    test = soup.prettify()
    trigger = "<div class=\"full-script\">"
    trigger2 = "</div"
    a = test.find(trigger)
    b = test[a:].find(trigger2)
    buffer = a+len(trigger)+1
    buffer2 = b+a
    test = test[buffer:buffer2]
    test = test.replace("       ","")
    test = test.replace("<br/>\n<br/>","")
    test = test.replace("\n<br/>\n"," ")

    c = test.find("<script async=\"\" src")
    while c != -1:
        d = test[c:].find(".push({});\n</script>")+len(".push({});\n</script>")+c
        test = test[:c] + test[d+2:]
        c = test.find("<script async=\"\" src")
    test = "\n".join(test.splitlines()[:-4])
    return test


r = requests.get('https://subslikescript.com/series/Suits-1632701', auth=('user', 'pass'))
soup = BeautifulSoup(r.text,"html.parser")

links = []

for link in soup.find_all('a'):
    links.append(link.get('href'))

links = links[5:-1]
links.pop(30)
print()
#print(links[62:])


f = open("pilot.txt", "a")
for link in links[links.index("/series/Suits-1632701/season-7/episode-1-Skin_in_the_Game"):]:
    print(link)
    f.write(link+"\n\n")
    f.write(grabScript("https://subslikescript.com"+link)+"\n\n")

f.close()