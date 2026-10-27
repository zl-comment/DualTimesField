from pathlib import Path
import uno,time
from com.sun.star.beans import PropertyValue
P=Path(__file__).parent
local=uno.getComponentContext();resolver=local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',local)
for i in range(30):
 try:
  ctx=resolver.resolve('uno:socket,host=localhost,port=2013;urp;StarOffice.ComponentContext');break
 except Exception:time.sleep(.5)
desktop=ctx.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop',ctx)
def prop(name,value):
 p=PropertyValue();p.Name=name;p.Value=value;return p
f=P/'电价尖峰预测研究_阶段报告_论文章节版.docx'
doc=desktop.loadComponentFromURL(uno.systemPathToFileUrl(str(f)),'_blank',0,(prop('Hidden',True),prop('UpdateDocMode',3)))
indexes=doc.getDocumentIndexes();print('INDEXES',indexes.getCount())
for i in range(indexes.getCount()):indexes.getByIndex(i).update()
doc.refresh()
for i in range(indexes.getCount()):indexes.getByIndex(i).update()
doc.store()
doc.storeToURL(uno.systemPathToFileUrl(str(f.with_suffix('.pdf'))),(prop('FilterName','writer_pdf_Export'),prop('Overwrite',True)))
print('DONE',f)
doc.close(True)
