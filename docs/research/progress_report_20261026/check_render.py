import fitz,json
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
O=Path(__file__).parent
p=fitz.open(O/'电价尖峰预测研究_阶段汇报_20261026.pdf')
print('PAGES',len(p))
report=[]
for i,page in enumerate(p):
 text=page.get_text();report.append({'page':i+1,'chars':len(text),'start':text[:100],'end':text[-120:]})
 print(i+1,len(text),text[40:100].replace('\n',' '))
 imgs=[]
for i,page in enumerate(p):
 pix=page.get_pixmap(matrix=fitz.Matrix(.45,.45));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im=ImageOps.expand(im,border=3,fill='#999999');im=ImageOps.expand(im,border=(0,20,0,0),fill='white');ImageDraw.Draw(im).text((8,4),str(i+1),fill='black');imgs.append(im)
for start in range(0,len(imgs),12):
 subset=imgs[start:start+12];w=max(x.width for x in subset);h=max(x.height for x in subset);canvas=Image.new('RGB',(w*4,h*3),'#cccccc')
 for j,im in enumerate(subset):canvas.paste(im,((j%4)*w,(j//4)*h))
 canvas.save(O/f'contact_{start//12+1}.png')
(O/'layout_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
