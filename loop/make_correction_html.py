"""
Generate a self-contained HTML window-correction tool for the remaining group
homes (GH4, GH5 on page 17). Embeds the plan crops (base64) + detector
predictions; runs fully in a browser, exports corrected boxes as JSON in the
project's page-absolute [ymin,xmin,ymax,xmax] format.

Run: PYTHONPATH=. python3 loop/make_correction_html.py
Output: loop/correct_windows.html
"""
import sys, base64, json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.services import stantec_detector_core as det  # noqa

P = ROOT / "projects" / "7dde299f-ec42-4424-8f46-f865bfdb4a3b"
PDF = P / "original.pdf"
IMG = P / "page_17.png"
GHS = ["4", "5"]
PAGE = 17


def build():
    page = cv2.imread(str(IMG))
    regions = {r["group_home"]: r for r in det.detect_stantec_plan_regions(PDF, PAGE, page_image_path=IMG)}
    data = {}
    for gh in GHS:
        r = regions[gh]
        y0, x0, y1, x1 = r["box_px"]
        crop = page[y0:y1, x0:x1]
        ok, buf = cv2.imencode(".png", crop)
        b64 = base64.b64encode(buf).decode()
        wins = det.detect_stantec_windows_for_page(PDF, PAGE, page_image_path=IMG, group_homes=[gh])
        boxes = []
        for i, w in enumerate(wins, 1):
            b = w["box_px"]  # page-absolute [ymin,xmin,ymax,xmax]
            boxes.append({"label": f"W-{i:02d}", "x": b[1] - x0, "y": b[0] - y0,
                          "w": b[3] - b[1], "h": b[2] - b[0]})
        data[f"GH{gh}"] = {"img": "data:image/png;base64," + b64,
                           "w": int(x1 - x0), "h": int(y1 - y0),
                           "cropOrigin": [int(x0), int(y0)], "page": PAGE,
                           "boxes": boxes}
    return data


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Window Correction — GH4 / GH5</title>
<style>
 body{margin:0;font-family:system-ui,Arial,sans-serif;background:#1e1e1e;color:#eee}
 #bar{position:sticky;top:0;background:#252526;padding:8px 12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #444;z-index:10}
 button{background:#0e639c;color:#fff;border:0;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:13px}
 button:hover{background:#1177bb}
 button.tab{background:#3a3d41}button.tab.on{background:#0e639c}
 button.danger{background:#a1260d}
 #wrap{overflow:auto;height:calc(100vh - 92px);background:#2d2d2d}
 canvas{display:block;cursor:crosshair}
 .hint{font-size:12px;color:#aaa}
 #info{font-size:13px}
 kbd{background:#444;border-radius:3px;padding:1px 5px;font-size:11px}
</style></head><body>
<div id="bar">
  <strong>Window Correction</strong>
  <span id="tabs"></span>
  <button onclick="zoomBy(1.25)">Zoom +</button>
  <button onclick="zoomBy(0.8)">Zoom −</button>
  <button onclick="fit()">Fit</button>
  <button class="danger" onclick="delSel()">Delete box <kbd>Del</kbd></button>
  <button onclick="exportJSON()">⬇ Export corrected JSON</button>
  <span id="info"></span>
  <span class="hint">Drag empty area = new box · click box to select · drag to move · drag corner to resize</span>
</div>
<div id="wrap"><canvas id="cv"></canvas></div>
<script>
const DATA = __DATA__;
let cur=Object.keys(DATA)[0], zoom=0.45, boxes=[], sel=-1;
let act=null, start=null, orig=null, corner=null;
const cv=document.getElementById('cv'), ctx=cv.getContext('2d'), wrap=document.getElementById('wrap');
const img=new Image();
function tabs(){let h='';for(const k of Object.keys(DATA))h+=`<button class="tab ${k===cur?'on':''}" onclick="load('${k}')">${k} (${DATA[k].boxes.length} det)</button>`;document.getElementById('tabs').innerHTML=h;}
function load(k){cur=k;boxes=DATA[k].boxes.map(b=>({...b}));sel=-1;tabs();img.src=DATA[k].img;}
img.onload=()=>{resize();draw();};
function resize(){cv.width=DATA[cur].w*zoom;cv.height=DATA[cur].h*zoom;}
function zoomBy(f){zoom=Math.max(0.1,Math.min(3,zoom*f));resize();draw();}
function fit(){zoom=(wrap.clientWidth-20)/DATA[cur].w;resize();draw();}
function draw(){
 ctx.clearRect(0,0,cv.width,cv.height);
 ctx.drawImage(img,0,0,cv.width,cv.height);
 boxes.forEach((b,i)=>{
  const x=b.x*zoom,y=b.y*zoom,w=b.w*zoom,h=b.h*zoom;
  ctx.lineWidth=i===sel?3:2;ctx.strokeStyle=i===sel?'#ff3b30':'#0a84ff';
  ctx.strokeRect(x,y,w,h);
  if(i===sel){ctx.fillStyle='#ff3b30';corners(b).forEach(c=>ctx.fillRect(c[0]*zoom-4,c[1]*zoom-4,8,8));}
 });
 document.getElementById('info').textContent=`${cur}: ${boxes.length} windows`+(sel>=0?` · selected ${boxes[sel].label}`:'');
}
function corners(b){return [[b.x,b.y],[b.x+b.w,b.y],[b.x,b.y+b.h],[b.x+b.w,b.y+b.h]];}
function ipos(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left)/zoom,(e.clientY-r.top)/zoom];}
function hitCorner(b,px,py){const c=corners(b);for(let i=0;i<4;i++){if(Math.abs(c[i][0]-px)<8/zoom&&Math.abs(c[i][1]-py)<8/zoom)return i;}return -1;}
function inBox(b,px,py){return px>=b.x&&px<=b.x+b.w&&py>=b.y&&py<=b.y+b.h;}
cv.addEventListener('mousedown',e=>{
 const [px,py]=ipos(e);
 if(sel>=0){const c=hitCorner(boxes[sel],px,py);if(c>=0){act='resize';corner=c;orig={...boxes[sel]};start=[px,py];return;}}
 for(let i=boxes.length-1;i>=0;i--){if(inBox(boxes[i],px,py)){sel=i;act='move';orig={...boxes[i]};start=[px,py];draw();return;}}
 sel=-1;act='draw';start=[px,py];boxes.push({label:nextLabel(),x:px,y:py,w:0,h:0});sel=boxes.length-1;draw();
});
window.addEventListener('mousemove',e=>{
 if(!act)return;const [px,py]=ipos(e);const dx=px-start[0],dy=py-start[1];const b=boxes[sel];
 if(act==='move'){b.x=orig.x+dx;b.y=orig.y+dy;}
 else if(act==='draw'){b.x=Math.min(start[0],px);b.y=Math.min(start[1],py);b.w=Math.abs(px-start[0]);b.h=Math.abs(py-start[1]);}
 else if(act==='resize'){let x1=orig.x,y1=orig.y,x2=orig.x+orig.w,y2=orig.y+orig.h;
   if(corner===0){x1=orig.x+dx;y1=orig.y+dy;}if(corner===1){x2=orig.x+orig.w+dx;y1=orig.y+dy;}
   if(corner===2){x1=orig.x+dx;y2=orig.y+orig.h+dy;}if(corner===3){x2=orig.x+orig.w+dx;y2=orig.y+orig.h+dy;}
   b.x=Math.min(x1,x2);b.y=Math.min(y1,y2);b.w=Math.abs(x2-x1);b.h=Math.abs(y2-y1);}
 draw();
});
window.addEventListener('mouseup',()=>{if(act==='draw'&&(boxes[sel].w<3||boxes[sel].h<3)){boxes.splice(sel,1);sel=-1;}act=null;draw();});
window.addEventListener('keydown',e=>{if((e.key==='Delete'||e.key==='Backspace')&&sel>=0){delSel();}});
function delSel(){if(sel>=0){boxes.splice(sel,1);sel=-1;draw();}}
function nextLabel(){let n=1;const used=new Set(boxes.map(b=>b.label));while(used.has('W-'+String(n).padStart(2,'0')))n++;return 'W-'+String(n).padStart(2,'0');}
function exportJSON(){
 const [ox,oy]=DATA[cur].cropOrigin;
 const out=boxes.map((b,i)=>({label:'W-'+String(i+1).padStart(2,'0'),
   box_px:[Math.round(b.y+oy),Math.round(b.x+ox),Math.round(b.y+b.h+oy),Math.round(b.x+b.w+ox)]}));
 const blob=new Blob([JSON.stringify({page:DATA[cur].page,group_home:cur,width:9362,height:6622,corrected_windows:out},null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`corrected_${cur}_page${DATA[cur].page}.json`;a.click();
}
load(cur);
</script></body></html>"""


def main():
    data = build()
    html = HTML.replace("__DATA__", json.dumps(data))
    out = ROOT / "loop" / "correct_windows.html"
    out.write_text(html, encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({mb:.1f} MB)  GHs={list(data.keys())} "
          f"boxes={[len(data[k]['boxes']) for k in data]}")


if __name__ == "__main__":
    main()
