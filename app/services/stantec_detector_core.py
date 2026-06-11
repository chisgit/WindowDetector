#!/usr/bin/env python3
"""Deterministic floor-plan window/opening detector with profile auto-selection.

v28: generic wall-band detector plus wider pane merging, relaxed hatch-safe vertical windows, and thin-interior-wall horizontal cap pass.

Key point: no per-window coordinate overrides. For Stantec A102-style drawings,
window boxes are derived from detected wall bands and cap/sash strokes:

  long wall rails -> paired wall bands -> cap strokes crossing the band ->
  consecutive cap pairs -> pane merge -> exterior-side filter -> annotated PNG/CSV

No AI calls, no ML models, no OCR model, no training.
"""
from __future__ import annotations

import argparse, csv, re, zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import fitz
import numpy as np

@dataclass
class Candidate:
    x:int; y:int; w:int; h:int
    orientation:str='horizontal'
    source:str='unknown'
    score:float=0.0
    exterior_side:str='unknown'
    @property
    def cx(self): return self.x + self.w/2
    @property
    def cy(self): return self.y + self.h/2

@dataclass
class LineAxis:
    pos:float
    intervals:list[tuple[int,int]]
    total:int


def parse_pages(spec:str, max_pages:int)->list[int]:
    pages=set()
    for part in spec.split(','):
        part=part.strip()
        if not part: continue
        if '-' in part:
            a,b=map(int,part.split('-',1)); pages.update(range(a,b+1))
        else:
            pages.add(int(part))
    return [p for p in sorted(pages) if 1<=p<=max_pages]


def pdf_text_sample(pdf_path:Path, pages:list[int]|None=None, max_pages:int=6)->str:
    doc=fitz.open(str(pdf_path))
    idxs=[p-1 for p in pages if 1<=p<=len(doc)] if pages else list(range(min(max_pages,len(doc))))
    if not pages and len(doc)>6: idxs.extend(range(max(0,len(doc)-4),len(doc)))
    return '\n'.join(doc[i].get_text('text') for i in sorted(set(idxs))).upper()


def auto_select_profile(pdf_path:Path, pages:list[int]|None=None)->tuple[str,str]:
    txt=pdf_text_sample(pdf_path,pages)
    if 'HOLIDAY INN EXPRESS' in txt or 'ACCENT DESIGN' in txt or 'GUESTROOM' in txt:
        return 'hotel_guestrooms','matched PDF text: Holiday Inn / Accent Design / Guestroom'
    if 'STANTEC' in txt or 'GROUP HOME' in txt or 'ALBERTA INFRASTRUCTURE' in txt or 'A101' in txt or 'A102' in txt:
        return 'stantec_group_homes','matched PDF text: Stantec / Group Home / Alberta Infrastructure / A101/A102'
    raise SystemExit('Could not auto-select profile confidently. Re-run with --profile.')


def render_page_dpi(pdf_path:Path, page_1_based:int, dpi:int=200)->tuple[np.ndarray,float,fitz.Rect]:
    doc=fitz.open(str(pdf_path)); page=doc[page_1_based-1]
    zoom=dpi/72.0
    pix=page.get_pixmap(matrix=fitz.Matrix(zoom,zoom), alpha=False)
    arr=np.frombuffer(pix.tobytes('png'),dtype=np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError('Could not render PDF page')
    return img, zoom, page.rect


def render_page_zoom(pdf_path:Path, page_1_based:int, zoom:float=1.0)->np.ndarray:
    doc=fitz.open(str(pdf_path)); page=doc[page_1_based-1]
    pix=page.get_pixmap(matrix=fitz.Matrix(zoom,zoom), alpha=False)
    arr=np.frombuffer(pix.tobytes('png'),dtype=np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError('Could not render PDF page')
    return img


def binarize(img:np.ndarray, threshold:int=210)->np.ndarray:
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)[1]


def density(mask:np.ndarray, x1:int, y1:int, x2:int, y2:int)->float:
    H,W=mask.shape[:2]
    x1=max(0,min(W,int(x1))); x2=max(0,min(W,int(x2)))
    y1=max(0,min(H,int(y1))); y2=max(0,min(H,int(y2)))
    if x2<=x1 or y2<=y1: return 0.0
    return float(mask[y1:y2,x1:x2].mean()/255.0)


def connected_components(mask:np.ndarray)->list[tuple[int,int,int,int,int]]:
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask,8)
    return [tuple(map(int,stats[i])) for i in range(1,n)]


def draw_boxes(img:np.ndarray, boxes:Iterable[Candidate], label_prefix:str='W')->np.ndarray:
    out=img.copy()
    for i,c in enumerate(boxes,1):
        color=(255,0,0) if c.orientation=='vertical' else (0,0,255)
        cv2.rectangle(out,(max(0,c.x-5),max(0,c.y-5)),(c.x+c.w+5,c.y+c.h+5),color,3)
        cv2.putText(out,f'{label_prefix}{i}',(c.x,max(18,c.y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2,cv2.LINE_AA)
    return out


def write_candidate_csv(path:Path, boxes:Iterable[Candidate], extra_cols:dict|None=None)->None:
    extra_cols=extra_cols or {}
    base=['id','x_px','y_px','width_px','height_px','center_x_px','center_y_px','orientation','source','score','exterior_side']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(list(extra_cols.keys())+base)
        for idx,c in enumerate(boxes,1):
            w.writerow(list(extra_cols.values())+[idx,c.x,c.y,c.w,c.h,round(c.cx,2),round(c.cy,2),c.orientation,c.source,round(c.score,5),c.exterior_side])

# ---------------- Stantec profile: title crop ----------------

def find_stantec_plan_titles(pdf_path:Path, page_1_based:int):
    page=fitz.open(str(pdf_path))[page_1_based-1]
    words=page.get_text('words')
    rows=[]
    for block_no in sorted({w[5] for w in words}):
        for line_no in sorted({w[6] for w in words if w[5]==block_no}):
            line=sorted([w for w in words if w[5]==block_no and w[6]==line_no], key=lambda w:w[0])
            txt=' '.join(w[4] for w in line).upper()
            m=re.search(r'(?:DEMO\s+FLOOR\s+PLAN|CONSTRUCTION\s+PLAN)\s*-\s*GROUP\s+HOME\s+(\d+)',txt)
            if m:
                bbox=(min(w[0] for w in line), min(w[1] for w in line), max(w[2] for w in line), max(w[3] for w in line))
                rows.append((m.group(1),txt,bbox))
    return rows, page.rect


def stantec_crop_from_title(bbox, page_rect, zoom:float)->tuple[int,int,int,int]:
    x0,y0,x1,y1=bbox
    crop_pdf=fitz.Rect(max(0,x0-170), max(0,y0-930), min(page_rect.width,x1+170), min(page_rect.height,y1+125))
    return tuple(int(round(v*zoom)) for v in (crop_pdf.x0,crop_pdf.y0,crop_pdf.x1,crop_pdf.y1))

# ---------------- Stantec profile: wall-band algorithm ----------------

def merge_intervals(intervals:list[tuple[int,int]], gap:int=18)->list[tuple[int,int]]:
    if not intervals: return []
    intervals=sorted(intervals)
    out=[[intervals[0][0], intervals[0][1]]]
    for a,b in intervals[1:]:
        if a<=out[-1][1]+gap: out[-1][1]=max(out[-1][1],b)
        else: out.append([a,b])
    return [tuple(x) for x in out]


def interval_overlap_len(a:tuple[int,int], b:tuple[int,int])->int:
    return max(0, min(a[1],b[1])-max(a[0],b[0]))


def cluster_line_axes(mask:np.ndarray, orient:str)->list[LineAxis]:
    comps=connected_components(mask)
    items=[]
    for x,y,w,h,a in comps:
        if orient=='vertical':
            if h>=70 and a>=50 and w<=20: items.append((x+w/2,(y,y+h)))
        else:
            if w>=70 and a>=50 and h<=20: items.append((y+h/2,(x,x+w)))
    items=sorted(items, key=lambda z:z[0])
    clusters=[]
    for pos,iv in items:
        if not clusters or abs(pos-clusters[-1][0][-1])>6:
            clusters.append(([pos],[iv]))
        else:
            clusters[-1][0].append(pos); clusters[-1][1].append(iv)
    axes=[]
    for ps,ivs in clusters:
        intervals=merge_intervals(ivs,18)
        total=sum(b-a for a,b in intervals)
        if total>=90:
            axes.append(LineAxis(float(np.mean(ps)), intervals, total))
    return axes


def pair_wall_axes(axes:list[LineAxis], max_pair_dist:int=42, min_axis_total:int=180)->list[tuple[LineAxis,LineAxis,float,int,tuple[int,int]]]:
    pairs=[]
    for i,a in enumerate(axes):
        for b in axes[i+1:]:
            dist=b.pos-a.pos
            if not (8<=dist<=max_pair_dist): continue
            ov=sum(interval_overlap_len(ia,ib) for ia in a.intervals for ib in b.intervals)
            span=(min(a.intervals[0][0],b.intervals[0][0]), max(a.intervals[-1][1],b.intervals[-1][1]))
            if min(a.total, b.total) >= min_axis_total and (ov>=80 or (a.total>250 and b.total>250 and span[1]-span[0]>200)):
                pairs.append((a,b,dist,ov,span))
    return pairs


def short_line_strokes(hmask:np.ndarray, vmask:np.ndarray):
    hst=[]
    for x,y,w,h,a in connected_components(hmask):
        if 8<=w<=150 and 1<=h<=10 and a>=8:
            hst.append((x,y,w,h,a,x+w/2,y+h/2))
    vst=[]
    for x,y,w,h,a in connected_components(vmask):
        if 1<=w<=10 and 8<=h<=150 and a>=8:
            vst.append((x,y,w,h,a,x+w/2,y+h/2))
    return hst, vst


def suppress_overlaps(cands:list[Candidate], tol:int=34)->list[Candidate]:
    # Prefer larger merged candidates; tie-break by score. No coordinate overrides.
    ordered=sorted(cands, key=lambda c:(-(c.w*c.h), -c.score))
    kept=[]
    for c in ordered:
        if any(c.orientation==k.orientation and abs(c.cx-k.cx)<tol and abs(c.cy-k.cy)<tol for k in kept):
            continue
        kept.append(c)
    return sorted(kept, key=lambda c:(c.y,c.x))


def merge_adjacent_panes(cands:list[Candidate])->list[Candidate]:
    # Merge two-pane windows only when the panes share the same wall band and touch/near-touch.
    out=[]
    for c in sorted(cands, key=lambda z:(z.orientation,z.x,z.y)):
        merged=False
        for k in out:
            if c.orientation!=k.orientation: continue
            if c.orientation=='vertical':
                same_band=abs(c.x-k.x)<=8 and abs(c.w-k.w)<=18
                gap=c.y-(k.y+k.h)
                combined_h=max(k.y+k.h,c.y+c.h)-min(k.y,c.y)
                # Bound pane merge so stacked separate room windows are not fused into one tall artifact.
                if same_band and -8<=gap<=28 and combined_h<=190:
                    y0=min(k.y,c.y); y1=max(k.y+k.h,c.y+c.h)
                    k.y=y0; k.h=y1-y0; k.score=max(k.score,c.score); k.source += '+pane_merge'; merged=True; break
            else:
                same_band=abs(c.y-k.y)<=8 and abs(c.h-k.h)<=18
                gap=c.x-(k.x+k.w)
                combined_w=max(k.x+k.w,c.x+c.w)-min(k.x,c.x)
                # Bound pane merge so adjacent separate top-wall windows are not fused into one long artifact.
                if same_band and -8<=gap<=46 and combined_w<=235:
                    x0=min(k.x,c.x); x1=max(k.x+k.w,c.x+c.w)
                    k.x=x0; k.w=x1-x0; k.score=max(k.score,c.score); k.source += '+pane_merge'; merged=True; break
        if not merged: out.append(c)
    return sorted(out, key=lambda c:(c.y,c.x))


def stantec_wall_band_windows(crop:np.ndarray, threshold:int=210, save_debug_dir:Path|None=None, text_mask:np.ndarray|None=None)->tuple[list[Candidate], list[Candidate]]:
    bw=binarize(crop, threshold)
    hmask=cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(8,1)))
    vmask=cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,8)))
    hlong=cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(70,1)))
    vlong=cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,70)))
    longmask=cv2.bitwise_or(hlong,vlong)
    hst, vst=short_line_strokes(hmask,vmask)
    vaxes=cluster_line_axes(vlong,'vertical')
    haxes=cluster_line_axes(hlong,'horizontal')
    vpairs=pair_wall_axes(vaxes, min_axis_total=180)
    hpairs=pair_wall_axes(haxes, min_axis_total=90)
    raw=[]

    # Vertical windows: two horizontal caps crossing a paired vertical wall band.
    for a,b,dist,ov,span in vpairs:
        left,right=a.pos,b.pos
        caps=[]
        for x,y,w,h,area,cx,cy in hst:
            if y<span[0]-30 or y>span[1]+30: continue
            if x<=left+8 and x+w>=right-8 and (right-left-8)<=w<=(right-left+80):
                caps.append((cy,x,y,w,h))
        groups=[]
        # Group cap strokes by the actual cap x/width so dimension ticks do not pair with window caps.
        for cy,x,y,w,h in sorted(caps,key=lambda c:(c[1],c[3],c[0])):
            for g in groups:
                if abs(x-g['x'])<=12 and abs(w-g['w'])<=20:
                    g['caps'].append((cy,x,y,w,h)); g['x']=0.8*g['x']+0.2*x; g['w']=0.8*g['w']+0.2*w; break
            else:
                groups.append({'x':float(x),'w':float(w),'caps':[(cy,x,y,w,h)]})
        for g in groups:
            ys=[]
            for cy,x,y,w,h in sorted(g['caps']):
                if not ys or abs(cy-ys[-1][0])>10: ys.append([cy,x,y,w,h])
            for j in range(len(ys)-1):
                _,x1,y1,w1,h1=ys[j]; _,x2,y2,w2,h2=ys[j+1]
                gap=ys[j+1][0]-ys[j][0]
                if 45<=gap<=130:
                    x0=int(round(max(min(x1,x2), left-6)))
                    x1b=int(round(min(max(x1+w1,x2+w2), right+8)))
                    y0=int(round(min(y1,y2))); y1b=int(round(max(y1+h1,y2+h2)))
                    left_long=density(longmask,x0-220,y0,x0-30,y1b)
                    right_long=density(longmask,x1b+30,y0,x1b+220,y1b)
                    local=density(bw,x0-20,y0-20,x1b+20,y1b+20)
                    rail=density(vmask,int(left)-5,y0,int(right)+5,y1b)
                    exterior=min(left_long,right_long)
                    if exterior<0.03 and local<0.35 and rail>0.08:
                        side='left' if left_long<right_long else 'right'
                        raw.append(Candidate(x0,y0,x1b-x0,y1b-y0,'vertical','derived_from_vertical_wall_band',2.0-exterior-local,side))

    # Horizontal windows: two vertical caps crossing a paired horizontal wall band.
    for a,b,dist,ov,span in hpairs:
        top,bottom=a.pos,b.pos
        caps=[]
        for x,y,w,h,area,cx,cy in vst:
            if x<span[0]-30 or x>span[1]+30: continue
            if y<=top+8 and y+h>=bottom-8 and (bottom-top-8)<=h<=(bottom-top+80):
                caps.append((cx,x,y,w,h))
        groups=[]
        # Group by cap y/height, so text strokes do not pair with wall-window caps.
        for cx,x,y,w,h in sorted(caps,key=lambda c:(c[2],c[4],c[0])):
            for g in groups:
                if abs(y-g['y'])<=12 and abs(h-g['h'])<=20:
                    g['caps'].append((cx,x,y,w,h)); g['y']=0.8*g['y']+0.2*y; g['h']=0.8*g['h']+0.2*h; break
            else:
                groups.append({'y':float(y),'h':float(h),'caps':[(cx,x,y,w,h)]})
        for g in groups:
            xs=[]
            for cx,x,y,w,h in sorted(g['caps']):
                if not xs or abs(cx-xs[-1][0])>10: xs.append([cx,x,y,w,h])
            for j in range(len(xs)-1):
                _,x1,y1,w1,h1=xs[j]; _,x2,y2,w2,h2=xs[j+1]
                gap=xs[j+1][0]-xs[j][0]
                if 45<=gap<=180:
                    x0=int(round(min(x1,x2))); x1b=int(round(max(x1+w1,x2+w2)))
                    y0=int(round(max(min(y1,y2), top-6))); y1b=int(round(min(max(y1+h1,y2+h2), bottom+8)))
                    above_long=density(longmask,x0,y0-220,x1b,y0-30)
                    below_long=density(longmask,x0,y1b+30,x1b,y1b+220)
                    local=density(bw,x0-20,y0-20,x1b+20,y1b+20)
                    rail=density(hmask,x0,int(top)-5,x1b,int(bottom)+5)
                    exterior=min(above_long,below_long)
                    if exterior<0.032 and local<0.35 and rail>0.08:
                        side='above' if above_long<below_long else 'below'
                        raw.append(Candidate(x0,y0,x1b-x0,y1b-y0,'horizontal','derived_from_horizontal_wall_band',2.0-exterior-local,side))

    # Generic pane merge, then size and overlap suppression. No per-window coordinates.
    merged=merge_adjacent_panes(raw)
    filtered=[]
    H,W=bw.shape[:2]
    for c in merged:
        # Remove the crop frame/dimension margin and title-band artifacts using only proportional crop bounds.
        # Retain legitimate edge windows near the north/east/west exterior walls while still
        # excluding crop/title/dimension bands proportionally.
        if c.y < int(0.09*H) or c.y+c.h > int(0.82*H):
            continue
        if c.x < int(0.13*W) or c.x+c.w > int(0.84*W):
            continue
        if c.orientation=='vertical' and not (10<=c.w<=55 and 50<=c.h<=125):
            # Doorways and spaces between stacked windows are taller than the actual cap/sash component.
            continue
        if c.orientation=='horizontal' and not (45<=c.w<=235 and 8<=c.h<=45):
            # Prevent very long adjacent-window spaces from merging into one false candidate, while allowing two-pane windows.
            continue

        # Use structural ink for door/space rejection. Native PDF text is removed from this test
        # so room tags/dimensions do not accidentally reject real windows.
        struct_bw = bw if text_mask is None else cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))
        if c.orientation == 'vertical':
            # Hatch-heavy demolition rooms put a lot of diagonal ink beside real exterior windows.
            # Do not reject those based on generic interior ink alone. Only reject very dense
            # door-like geometry in the immediate interior strip, and only when the candidate is
            # taller than a normal single-pane cap.
            if c.exterior_side == 'left':
                interior_ink = density(struct_bw, c.x+c.w, c.y-35, c.x+c.w+90, c.y+c.h+35)
            else:
                interior_ink = density(struct_bw, c.x-90, c.y-35, c.x, c.y+c.h+35)
            if False and c.h > 105 and interior_ink > 0.18:
                continue
        else:
            # Door threshold rejection: if the interior-side central strip has a long vertical jamb/leaf
            # crossing it, this is a door/opening, not a window cap.
            if c.exterior_side == 'above':
                iy1, iy2 = c.y+c.h+5, c.y+c.h+105
            else:
                iy1, iy2 = c.y-105, c.y-5
            jamb_ink = density(vmask, c.x + int(0.35*c.w), iy1, c.x + int(0.65*c.w), iy2)
            if jamb_ink > 0.02:
                continue

        # Deterministic text/annotation suppression: window boxes should not overlap native PDF words.
        # This removes room tags, finish tags, dimension labels, and note bubbles without OCR or AI.
        if text_mask is not None and density(text_mask, c.x-4, c.y-4, c.x+c.w+4, c.y+c.h+4) > 0.08:
            continue
        filtered.append(c)
    filtered=suppress_overlaps(filtered, tol=34)

    if save_debug_dir:
        save_debug_dir.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(save_debug_dir/'debug_binarized.png'),bw)
        cv2.imwrite(str(save_debug_dir/'debug_long_walls.png'),longmask)
        cv2.imwrite(str(save_debug_dir/'debug_raw_wall_band_candidates.png'),draw_boxes(crop,raw,'R'))
    return filtered, raw

# Minimal hotel profile kept from prior version: currently no changes.
def run_hotel(pdf:Path, pages:list[int], outdir:Path, zoom:float, ink_threshold:int, save_raw:bool):
    raise SystemExit('Hotel profile not included in v7 wall-band test build. Use v5/v6 for hotel pages or merge this Stantec function into the full script.')



def crop_text_mask(pdf:Path, page_num:int, zoom:float, crop_rect_px:tuple[int,int,int,int], crop_shape:tuple[int,int,int])->np.ndarray:
    page=fitz.open(str(pdf))[page_num-1]
    x0,y0,x1,y1=crop_rect_px
    mask=np.zeros(crop_shape[:2], dtype=np.uint8)
    for wx0,wy0,wx1,wy1,word,*rest in page.get_text('words'):
        # Ignore single-character keynote numbers inside circles/triangles less aggressively; they are handled by geometry.
        px0=int(round(wx0*zoom))-x0; py0=int(round(wy0*zoom))-y0
        px1=int(round(wx1*zoom))-x0; py1=int(round(wy1*zoom))-y0
        if px1<0 or py1<0 or px0>=mask.shape[1] or py0>=mask.shape[0]:
            continue
        pad=5
        cv2.rectangle(mask,(max(0,px0-pad),max(0,py0-pad)),(min(mask.shape[1]-1,px1+pad),min(mask.shape[0]-1,py1+pad)),255,-1)
    return mask


def rect_iou(a:Candidate,b:Candidate)->float:
    ax1,ay1,ax2,ay2=a.x,a.y,a.x+a.w,a.y+a.h
    bx1,by1,bx2,by2=b.x,b.y,b.x+b.w,b.y+b.h
    ix=max(0,min(ax2,bx2)-max(ax1,bx1)); iy=max(0,min(ay2,by2)-max(ay1,by1))
    inter=ix*iy
    if inter<=0: return 0.0
    area=a.w*a.h+b.w*b.h-inter
    return inter/area if area else 0.0


def detect_generic_interior_thin_horizontal_caps(crop:np.ndarray, text_mask:np.ndarray|None=None)->list[Candidate]:
    """Deterministically detect horizontal single-pane window caps carried by thin
    interior wall linework, which are not always captured as paired thick wall-bands.

    This is not a per-window coordinate override. It scans the plan interior for
    long shallow horizontal cap components, suppresses native PDF text/dimensions,
    rejects components in exterior dimension/title bands, and requires nearby
    thin-wall horizontal continuity.
    """
    gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw=cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    if text_mask is not None:
        # Remove text before extracting long cap strokes.
        bw=cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))
    H,W=bw.shape[:2]
    # Long horizontal strokes that can be sash/cap on thin walls.
    hor=cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(46,3)))
    comps=connected_components(hor)
    cands=[]
    for x,y,w,h,a in comps:
        if not (75 <= w <= 230 and 4 <= h <= 28):
            continue
        # Stay inside plan area, away from title and dimension bands.
        if y < int(0.09*H) or y+h > int(0.82*H) or x < int(0.13*W) or x+w > int(0.84*W):
            continue
        if text_mask is not None and density(text_mask,x-5,y-5,x+w+5,y+h+5)>0.02:
            continue
        local=density(bw,x-8,y-8,x+w+8,y+h+8)
        if local < 0.05 or local > 0.45:
            continue
        # Require thin-wall context: either a second close parallel horizontal stroke,
        # or a nearby long wall segment continuing left/right at the same y-band.
        above=density(hor,x-10,y-24,x+w+10,y-6)
        below=density(hor,x-10,y+h+6,x+w+10,y+h+24)
        left_cont=density(hor,x-100,y-6,x-10,y+h+6)
        right_cont=density(hor,x+w+10,y-6,x+w+100,y+h+6)
        if max(above,below,left_cont,right_cont) < 0.015:
            continue
        cands.append(Candidate(int(x),int(y),int(w),int(h),'horizontal','generic_interior_thin_wall_cap',1.0+local,'interior'))
    # Remove duplicates among cap components.
    return suppress_overlaps(cands, tol=28)



def trim_candidate_to_cap_ink(crop:np.ndarray, c:Candidate, text_mask:np.ndarray|None=None)->Candidate|None:
    """Trim a candidate to the compact sash/cap ink inside its wall-band box.

    This is a generic correction for cases where wall-band pairing returns the
    empty wall/opening space adjacent to the actual drawn cap. It uses only ink
    projection inside/near the candidate, not room labels or coordinates.
    """
    gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw=cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    if text_mask is not None:
        bw=cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))
    H,W=bw.shape[:2]
    pad=8
    x1=max(0,c.x-pad); y1=max(0,c.y-pad); x2=min(W,c.x+c.w+pad); y2=min(H,c.y+c.h+pad)
    roi=bw[y1:y2,x1:x2]
    if roi.size == 0:
        return None

    if c.orientation == 'horizontal':
        proj=roi.sum(axis=0)/255.0
        thresh=max(2.0, roi.shape[0]*0.18)
        inds=np.where(proj>=thresh)[0]
        if len(inds)==0:
            return None
        segs=[]
        st=prev=int(inds[0])
        for val0 in inds[1:]:
            val=int(val0)
            if val>prev+3:
                segs.append((st,prev)); st=val
            prev=val
        segs.append((st,prev))
        # Space-after-window correction. Build compact cap regions from ink
        # projections. A true window cap is usually the dense/continuous segment;
        # the false space-after-window is the loose cluster of short ticks after it.
        total_width = c.w + 2*pad
        regions=[]
        cur=[segs[0][0], segs[0][1]]
        for a,b in segs[1:]:
            if a-cur[1] > 18:
                regions.append(cur); cur=[a,b]
            else:
                cur[1]=b
        regions.append(cur)

        scored=[]
        for a,b in regions:
            width_i=b-a+1
            ink=float(proj[a:b+1].sum())
            density_i=ink/max(1,width_i)
            # Penalize loose tick clusters; reward compact dense cap segments.
            score=ink + 20*density_i - 0.15*max(0,width_i-170)
            scored.append((score,a,b,width_i,ink,density_i))

        # If the first region is a long dense cap and later regions are loose small
        # ticks, keep the first region. This fixes "box on the space after the window".
        # First, use the original projection segments to catch the common case:
        # a dense left/leading cap segment followed by a series of small wall-space ticks.
        if len(segs) >= 2:
            s0_len = segs[0][1] - segs[0][0] + 1
            s0_ink = float(proj[segs[0][0]:segs[0][1]+1].sum())
            later_a = segs[1][0]; later_b = segs[-1][1]
            later_width = later_b - later_a + 1
            later_ink = float(proj[later_a:later_b+1].sum())
            if 45 <= s0_len <= 170 and s0_ink/max(1,s0_len) >= later_ink/max(1,later_width)*0.85:
                best=[segs[0][0], segs[0][1]]
            elif len(regions) >= 2:
                first= scored[0]
                later_a=regions[1][0]; later_b=regions[-1][1]
                later_width=later_b-later_a+1
                later_ink=float(proj[later_a:later_b+1].sum())
                first_len=first[3]
                first_density=first[5]
                later_density=later_ink/max(1,later_width)
                if 45 <= first_len <= 170 and first_density >= later_density*0.85:
                    best=[regions[0][0], regions[0][1]]
                elif 45 <= later_width <= 170:
                    best=[later_a,later_b]
                else:
                    best=list(max(scored, key=lambda t:t[0])[1:3])
            else:
                best=list(max(scored, key=lambda t:t[0])[1:3])
        else:
            best=list(max(scored, key=lambda t:t[0])[1:3])

        width=best[1]-best[0]+1
        if width > 170 and len(segs) > 1:
            sub=[]
            cur=[segs[0][0],segs[0][1]]
            for a,b in segs[1:]:
                if a-cur[1] > 14:
                    sub.append(cur); cur=[a,b]
                else:
                    cur[1]=b
            sub.append(cur)
            valid=[ab for ab in sub if 45 <= ab[1]-ab[0]+1 <= 170]
            if valid:
                best=max(valid, key=lambda ab: float(proj[ab[0]:ab[1]+1].sum()))
        gx1=x1+best[0]; gx2=x1+best[1]+1
        # Keep vertical extent close to original wall band.
        new=Candidate(int(gx1), int(c.y), int(max(1,gx2-gx1)), int(c.h), c.orientation, c.source+'+cap_ink_trim', c.score, c.exterior_side)
        if new.w < 45 or new.w > 210:
            return c
        return new

    else:
        proj=roi.sum(axis=1)/255.0
        thresh=max(2.0, roi.shape[1]*0.18)
        inds=np.where(proj>=thresh)[0]
        if len(inds)==0:
            return None
        segs=[]
        st=prev=int(inds[0])
        for val0 in inds[1:]:
            val=int(val0)
            if val>prev+3:
                segs.append((st,prev)); st=val
            prev=val
        segs.append((st,prev))

        # Reject "space after window" / door-like false positives:
        # true vertical window caps have two or more meaningful horizontal cap bands,
        # with a normal window height. Very sparse/tall gaps are not windows.
        meaningful=[(a,b) for a,b in segs if b-a+1 >= 4]
        if c.h > 95 and len(meaningful) <= 2:
            return None

        # Trim to the tight span between the first and last meaningful cap bands,
        # bounded to normal single/two-pane window height.
        if len(meaningful) >= 2:
            gy1=y1+meaningful[0][0]
            gy2=y1+meaningful[-1][1]+1
            nh=gy2-gy1
            if 45 <= nh <= 125:
                new=Candidate(int(c.x), int(gy1), int(c.w), int(nh), c.orientation, c.source+'+cap_ink_trim', c.score, c.exterior_side)
                return new
        return c


def reject_space_after_window_artifacts(crop:np.ndarray, boxes:list[Candidate], text_mask:np.ndarray|None=None)->list[Candidate]:
    """Remove candidates that represent empty wall/opening space after the real window.

    Generic rules:
    - same wall-band adjacent vertical candidates: keep the normal cap, drop the
      overlapping/stacked extension space.
    - very narrow/tall candidates with weak central cap ink are hatch/space artifacts.
    - candidates with one almost-empty side rail are not true between-wall caps.
    """
    gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw=cv2.threshold(gray,245,255,cv2.THRESH_BINARY_INV)[1]
    if text_mask is not None:
        bw=cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))
    keep=[True]*len(boxes)

    # Drop overlapping/stacked vertical extensions on the same wall band.
    for i,a in enumerate(boxes):
        if a.orientation!='vertical' or not keep[i]: continue
        for j,b in enumerate(boxes):
            if i==j or b.orientation!='vertical' or not keep[j]: continue
            if abs(a.cx-b.cx) <= 34:
                overlap=max(0, min(a.y+a.h,b.y+b.h)-max(a.y,b.y))
                touch=min(abs(a.y-(b.y+b.h)), abs(b.y-(a.y+a.h)))
                if overlap>0 or touch<=14:
                    # If one is substantially taller, it is usually the extension/opening space.
                    if a.h > 115 and a.h > b.h*1.25:
                        keep[i]=False
                    elif b.h > 115 and b.h > a.h*1.25:
                        keep[j]=False

    out=[]
    for k,c in enumerate(boxes):
        if not keep[k]:
            continue
        if c.orientation=='vertical':
            x,y,w,h=c.x,c.y,c.w,c.h
            roi=bw[max(0,y-5):y+h+5, max(0,x-6):x+w+6]
            if roi.size:
                left=roi[:,0:min(8,roi.shape[1])].mean()/255.0
                right=roi[:,max(0,roi.shape[1]-8):].mean()/255.0
                center=roi[:,min(8,roi.shape[1]//2):max(min(8,roi.shape[1]//2)+1,roi.shape[1]-8)].mean()/255.0 if roi.shape[1]>16 else roi.mean()/255.0
                # One side rail missing = likely space/door jamb after the actual cap.
                if min(left,right) < 0.07 and h > 110:
                    continue
                # Narrow/tall weak-center artifacts are usually hatch/space, not the cap.
                if w <= 18 and h > 100 and center < 0.16:
                    continue
                if w < 18 and h < 65:
                    continue
        out.append(c)
    return out



def split_long_vertical_raw_to_upper_caps(crop:np.ndarray, raw:list[Candidate], existing:list[Candidate], text_mask:np.ndarray|None=None)->list[Candidate]:
    """Generic recovery for missed vertical windows that sit above a door/wall-space.

    Some raw wall-band candidates merge a real upper window cap with lower wall/door
    space. This function splits only normal narrow vertical raw candidates by ink
    projection and keeps the upper compact cap portion when it has cap-band evidence.
    """
    gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw=cv2.threshold(gray,245,255,cv2.THRESH_BINARY_INV)[1]
    if text_mask is not None:
        bw=cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))
    recovered=[]
    for r in raw:
        if r.orientation!='vertical':
            continue
        if not (10 <= r.w <= 30 and 126 <= r.h <= 200):
            continue
        if text_mask is not None and density(text_mask, r.x-4, r.y-4, r.x+r.w+4, r.y+r.h+4) > 0.08:
            continue
        x,y,w,h=r.x,r.y,r.w,r.h
        roi=bw[max(0,y-6):y+h+6, max(0,x-6):x+w+6]
        if roi.size==0:
            continue
        proj=roi.sum(axis=1)/255.0
        inds=np.where(proj>=max(2.0, roi.shape[1]*0.18))[0]
        if len(inds)==0:
            continue
        bands=[]
        st=prev=int(inds[0])
        for val0 in inds[1:]:
            val=int(val0)
            if val>prev+3:
                bands.append((st,prev)); st=val
            prev=val
        bands.append((st,prev))
        meaningful=[(a,b) for a,b in bands if b-a+1>=4]
        if len(meaningful) < 3:
            continue
        # upper compact section from first three/four cap bands, capped to normal window height
        top=meaningful[0][0]
        candidate_bottom=meaningful[min(len(meaningful)-1,3)][1]+1
        gy1=max(0, y-6+top)
        gy2=max(gy1+1, min(y+h, y-6+candidate_bottom))
        nh=gy2-gy1
        if nh < 65:
            gy2=min(y+h, gy1+82); nh=gy2-gy1
        if not (65 <= nh <= 105):
            continue
        cand=Candidate(int(x), int(gy1), int(w), int(nh), 'vertical', r.source+'+split_upper_cap', r.score, r.exterior_side)
        # A true upper cap normally sits in continuous wall/detail ink both
        # above and below. Door openings/wall gaps often have the upper bars but
        # weak lower context, which made them look like short vertical windows.
        above_context=density(bw,x-40,gy1-100,x+w+40,gy1-10)
        below_context=density(bw,x-40,gy2+10,x+w+40,gy2+100)
        if min(above_context,below_context) < 0.055:
            continue
        # Do not duplicate existing windows on same wall band.
        if any(rect_iou(cand,e)>0.08 or (abs(cand.cx-e.cx)<28 and abs(cand.cy-e.cy)<35) for e in existing+recovered):
            continue
        recovered.append(cand)
    return recovered

def promote_upper_component_and_reject_wall_space(crop:np.ndarray, boxes:list[Candidate], raw:list[Candidate], text_mask:np.ndarray|None=None)->list[Candidate]:
    """Generic clean-up for vertical wall-band candidates.

    Problem pattern:
    - the detector sometimes selects a lower blank wall-space segment after a real
      vertical window cap; the actual cap is directly above on the same wall band.
    - dining/kitchen door/wall-space geometry can look like a short vertical cap.

    Rules are geometric:
    1) On the same wall band, if a current candidate is the lower part of a larger
       raw wall-band candidate, replace it with the upper normal-height component.
    2) Reject candidates that are isolated wall spaces without cap-pair evidence.
    """
    gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw=cv2.threshold(gray,245,255,cv2.THRESH_BINARY_INV)[1]
    if text_mask is not None:
        bw=cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))

    def cap_band_count(c:Candidate)->int:
        x,y,w,h=c.x,c.y,c.w,c.h
        roi=bw[max(0,y-6):y+h+6, max(0,x-6):x+w+6]
        if roi.size == 0: return 0
        proj=roi.sum(axis=1)/255.0
        inds=np.where(proj>=max(2.0, roi.shape[1]*0.18))[0]
        if len(inds)==0: return 0
        bands=[]
        st=prev=int(inds[0])
        for val0 in inds[1:]:
            val=int(val0)
            if val>prev+3:
                bands.append((st,prev)); st=val
            prev=val
        bands.append((st,prev))
        return sum(1 for a,b in bands if b-a+1>=4)

    out=[]
    for c in boxes:
        if c.orientation != 'vertical':
            out.append(c); continue

        replacement=None
        # Look for a larger raw candidate on same wall band that starts above c
        # and contains c. The true window cap is the upper normal-height segment.
        for r in raw:
            if r.orientation!='vertical': continue
            if c.exterior_side != 'left': continue
            if abs(r.cx-c.cx) > 14: continue
            if r.y < c.y-30 and r.y+r.h >= c.y+c.h-10 and r.h > c.h*1.35:
                upper_h = min(92, max(70, c.h))
                upper = Candidate(int(r.x), int(r.y), int(r.w), int(upper_h), c.orientation, c.source+'+upper_component_promote', c.score, c.exterior_side)
                if cap_band_count(upper) >= 2:
                    replacement=upper
                    break

        if replacement is not None:
            c=replacement

        # Reject isolated vertical wall spaces: a legitimate vertical window cap has
        # multiple horizontal cap bands. Door/wall-space segments often have only
        # one/two meaningful bands or missing side-rail balance.
        bands=cap_band_count(c)
        if c.h >= 88 and bands < 3:
            continue
        if c.h > 100 and bands < 4:
            continue

        out.append(c)

    return suppress_overlaps(out, tol=28)



def horizontal_pane_segments_from_ink(crop:np.ndarray, c:Candidate, text_mask:np.ndarray|None=None)->list[tuple[int,int]]:
    """Return compact x-ranges that have real horizontal pane/sash line evidence.

    Wall space tends to have only empty span or tiny ticks; a real window has a
    long internal parallel line segment inside the wall band. This is geometric
    and repeatable.
    """
    gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw=cv2.threshold(gray,245,255,cv2.THRESH_BINARY_INV)[1]
    if text_mask is not None:
        bw=cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))
    H,W=bw.shape[:2]
    x=max(0,c.x); y=max(0,c.y); w=max(1,c.w); h=max(1,c.h)
    roi=bw[y:min(H,y+h), x:min(W,x+w)]
    if roi.size == 0:
        return []
    # Internal horizontal sash evidence, excluding the outer wall rails.
    y1=max(1,h//3); y2=min(h, h*2//3+1)
    mid=roi[y1:y2,:]
    if mid.size == 0:
        return []
    proj=mid.sum(axis=0)/255.0
    thresh=max(1.0, mid.shape[0]*0.25)
    inds=np.where(proj>=thresh)[0]
    if len(inds)==0:
        return []
    segs=[]
    st=prev=int(inds[0])
    for val0 in inds[1:]:
        val=int(val0)
        if val>prev+2:
            segs.append((st,prev)); st=val
        prev=val
    segs.append((st,prev))

    # Keep substantial pane/sash runs separate. Tiny ticks are ignored; they are
    # usually wall-space markers or jamb fragments, not a full pane.
    panes=[]
    for a,b in segs:
        ww=b-a+1
        if ww >= 38:
            panes.append((int(x+a), int(x+b+1)))
    return panes


def horizontal_row_band_count(crop:np.ndarray, c:Candidate, text_mask:np.ndarray|None=None)->int:
    gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw=cv2.threshold(gray,245,255,cv2.THRESH_BINARY_INV)[1]
    if text_mask is not None:
        bw=cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))
    H,W=bw.shape[:2]
    roi=bw[max(0,c.y-6):min(H,c.y+c.h+6), max(0,c.x):min(W,c.x+c.w)]
    if roi.size == 0:
        return 0
    proj=roi.sum(axis=1)/255.0
    thresh=max(2.0, roi.shape[1]*0.25)
    inds=np.where(proj>=thresh)[0]
    if len(inds)==0:
        return 0
    bands=[]
    st=prev=int(inds[0])
    for val0 in inds[1:]:
        val=int(val0)
        if val>prev+2:
            bands.append((st,prev)); st=val
        prev=val
    bands.append((st,prev))
    return sum(1 for a,b in bands if b-a+1>=1)


def has_two_horizontal_panes(crop:np.ndarray, c:Candidate, text_mask:np.ndarray|None=None)->bool:
    panes=horizontal_pane_segments_from_ink(crop,c,text_mask)
    if len(panes) < 2:
        return False
    # Need two substantial panes of comparable width, adjacent on same wall band.
    widths=[b-a for a,b in panes]
    for i in range(len(panes)-1):
        a1,b1=panes[i]; a2,b2=panes[i+1]
        w1=b1-a1; w2=b2-a2
        gap=a2-b1
        if 55 <= w1 <= 95 and 55 <= w2 <= 95 and -4 <= gap <= 12 and min(w1,w2)/max(w1,w2) >= 0.72:
            return True
    return False


def trim_horizontal_to_pane_evidence(crop:np.ndarray, c:Candidate, text_mask:np.ndarray|None=None)->Candidate|None:
    if c.orientation != 'horizontal':
        return c
    panes=horizontal_pane_segments_from_ink(crop,c,text_mask)
    # No real internal parallel sash evidence -> wall space, not a window.
    if not panes:
        return None
    if horizontal_row_band_count(crop,c,text_mask) < 3:
        return None
    # If two adjacent valid panes exist, keep both. Otherwise keep the single
    # substantial pane and drop wall space before/after it.
    best_pair=None
    for i in range(len(panes)-1):
        a1,b1=panes[i]; a2,b2=panes[i+1]
        w1=b1-a1; w2=b2-a2; gap=a2-b1
        if 55 <= w1 <= 95 and 55 <= w2 <= 95 and -4 <= gap <= 12:
            best_pair=(a1,b2); break
    if best_pair:
        nx1,nx2=best_pair
    else:
        # Pick the longest true pane segment, not the loose wall-space ticks.
        nx1,nx2=max(panes, key=lambda ab: ab[1]-ab[0])
    # add a small cap margin
    nx1=max(0,nx1-4); nx2=nx2+4
    nw=nx2-nx1
    if nw < 45 or nw > 185:
        return None
    return Candidate(int(nx1), int(c.y), int(nw), int(c.h), c.orientation, c.source+'+pane_evidence_trim', c.score, c.exterior_side)


def _is_duplicate_candidate(candidate:Candidate, existing:list[Candidate], center_tol_x:int=35, center_tol_y:int=14, iou_threshold:float=0.08)->bool:
    for other in existing:
        if candidate.orientation != other.orientation:
            continue
        if rect_iou(candidate, other) > iou_threshold:
            return True
        if abs(candidate.cx-other.cx) < center_tol_x and abs(candidate.cy-other.cy) < center_tol_y:
            return True
    return False


def _near_vertical_raw_door_jamb(candidate:Candidate, raw:list[Candidate])->bool:
    """
    Compact horizontal strokes can appear inside door openings. If a raw vertical
    wall-band candidate passes through the compact stroke, treat it as door/jamb
    context and do not recover it as a window.
    """
    for v in raw:
        if v.orientation != 'vertical':
            continue
        vertical_axis_in_span = candidate.x - 10 <= v.cx <= candidate.x + candidate.w + 10
        y_overlap = min(candidate.y+candidate.h+12, v.y+v.h) - max(candidate.y-12, v.y)
        if vertical_axis_in_span and y_overlap > 0:
            return True
    return False


def recover_compact_horizontal_cap_windows(crop:np.ndarray, raw:list[Candidate], existing:list[Candidate], text_mask:np.ndarray|None=None)->list[Candidate]:
    """
    Recover compact single-panel horizontal window symbols that are valid wall
    caps but lack the longer internal pane runs used by two-panel cleanup.

    This stays generic: candidates must be compact, non-text, have multiple
    horizontal cap bands, avoid raw vertical jamb context, and not duplicate
    an existing horizontal detection.
    """
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    bw=cv2.threshold(gray,245,255,cv2.THRESH_BINARY_INV)[1]
    if text_mask is not None:
        bw=cv2.bitwise_and(bw, cv2.bitwise_not(text_mask))

    recovered:list[Candidate]=[]

    for r in raw:
        if r.orientation != 'horizontal':
            continue
        if not (55 <= r.w <= 115 and 8 <= r.h <= 24):
            continue
        if text_mask is not None and density(text_mask,r.x-4,r.y-4,r.x+r.w+4,r.y+r.h+4) > 0.02:
            continue
        if horizontal_row_band_count(crop,r,text_mask) < 3:
            continue
        if _near_vertical_raw_door_jamb(r,raw):
            continue

        t=trim_horizontal_to_pane_evidence(crop,r,text_mask)
        cand=t if t is not None else Candidate(r.x,r.y,r.w,r.h,'horizontal',r.source+'+compact_cap_recovery',r.score,r.exterior_side)
        if cand.w < 45 or cand.w > 120:
            continue
        if not _is_duplicate_candidate(cand, existing+recovered):
            recovered.append(cand)

    hor=cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(30,1)))
    for x,y,w,h,a in connected_components(hor):
        if not (55 <= w <= 115 and 1 <= h <= 5 and a >= 55):
            continue
        cand=Candidate(max(0,x-8), max(0,y-7), int(w+16), 16, 'horizontal', 'horizontal_component_cap_recovery', 1.0, 'unknown')
        if text_mask is not None and density(text_mask,cand.x-4,cand.y-4,cand.x+cand.w+4,cand.y+cand.h+4) > 0.02:
            continue
        if horizontal_row_band_count(crop,cand,text_mask) < 3:
            continue
        if not horizontal_pane_segments_from_ink(crop,cand,text_mask):
            continue
        if _near_vertical_raw_door_jamb(cand,raw):
            continue

        t=trim_horizontal_to_pane_evidence(crop,cand,text_mask)
        if t is None:
            continue
        t.source=cand.source+'+pane_evidence_trim'
        if 45 <= t.w <= 120 and not _is_duplicate_candidate(t, existing+recovered):
            recovered.append(t)

    return recovered


def refine_horizontal_two_pane_from_raw(crop:np.ndarray, filtered:list[Candidate], raw:list[Candidate], text_mask:np.ndarray|None=None)->list[Candidate]:
    """Generic two-pane horizontal window cleanup using raw wall-band candidates.

    Fixes two repeatable patterns:
    - cap trimming shrinks a valid two-pane window to only one pane
    - a wide candidate starts in wall/door space before the actual adjacent panes

    Uses only raw same-row cap candidates; no room-specific coordinates.
    """
    out=list(filtered)

    # 1) Restore valid raw two-pane candidates when a trimmed final candidate sits
    # inside it on the same wall band.
    for i,c in enumerate(out):
        if c.orientation != 'horizontal':
            continue
        best=None
        for r in raw:
            if r.orientation!='horizontal':
                continue
            if abs(r.cy-c.cy) > 8:
                continue
            if not (120 <= r.w <= 180 and 8 <= r.h <= 28):
                continue
            if not has_two_horizontal_panes(crop, r, text_mask):
                continue
            # r contains c or clearly overlaps it
            overlap=max(0, min(c.x+c.w, r.x+r.w)-max(c.x,r.x))
            if overlap >= min(c.w,r.w)*0.45 and c.w < r.w*0.82:
                if best is None or r.w > best.w:
                    best=r
        if best is not None:
            out[i]=Candidate(best.x,best.y,best.w,best.h,'horizontal',c.source+'+restore_raw_two_pane',c.score,c.exterior_side)

    # 2) Merge adjacent single-pane raw caps into one two-pane candidate.
    singles=[r for r in raw if r.orientation=='horizontal' and 55 <= r.w <= 105 and 8 <= r.h <= 28]
    additions=[]
    for i,a in enumerate(singles):
        for b in singles[i+1:]:
            if abs(a.cy-b.cy)>8:
                continue
            left,right=(a,b) if a.x<=b.x else (b,a)
            gap=b.x-(a.x+a.w) if a.x<=b.x else a.x-(b.x+b.w)
            union_x=min(a.x,b.x); union_y=min(a.y,b.y)
            union_w=max(a.x+a.w,b.x+b.w)-union_x
            union_h=max(a.y+a.h,b.y+b.h)-union_y
            if -12 <= gap <= 18 and 120 <= union_w <= 185 and union_h <= 32:
                cand=Candidate(int(union_x),int(union_y),int(union_w),int(union_h),'horizontal','raw_adjacent_two_pane_merge',max(a.score,b.score),'below')
                # Don't add duplicates, and require actual two-pane internal parallel-line evidence.
                if has_two_horizontal_panes(crop, cand, text_mask) and not any(abs(cand.cx-o.cx)<35 and abs(cand.cy-o.cy)<14 for o in out+additions if o.orientation=='horizontal'):
                    additions.append(cand)
    out.extend(additions)

    # 3) Remove wide wall-space candidates when a tighter two-pane candidate exists
    # inside its right/left side on the same row. This catches "space before the window".
    keep=[]
    for c in out:
        if c.orientation=='horizontal' and c.w > 185:
            has_tighter=False
            for o in out:
                if o is c or o.orientation!='horizontal':
                    continue
                if abs(o.cy-c.cy)>10 or not (120<=o.w<=185):
                    continue
                overlap=max(0,min(c.x+c.w,o.x+o.w)-max(c.x,o.x))
                # Tighter two-pane inside wide candidate -> wide one is wall space + window.
                if overlap >= o.w*0.5 and (o.x > c.x+25 or o.x+o.w < c.x+c.w-25):
                    has_tighter=True
                    break
            if has_tighter:
                continue
        keep.append(c)
    # Apply pane-evidence trim to all horizontal candidates: keep only the
    # segment(s) with internal parallel sash lines; wall space has no pane evidence.
    pane_trimmed=[]
    for c in keep:
        if c.orientation=='horizontal':
            t=trim_horizontal_to_pane_evidence(crop,c,text_mask)
            if t is not None:
                pane_trimmed.append(t)
        else:
            pane_trimmed.append(c)
    keep=pane_trimmed

    # Restore/add raw horizontal candidates only when the raw geometry contains
    # real internal pane/sash evidence. This recovers single-pane right-side
    # windows while still rejecting wall-space candidates that lack parallel
    # pane lines.
    for r in raw:
        if r.orientation!='horizontal':
            continue
        t=None
        if 120 <= r.w <= 185 and has_two_horizontal_panes(crop, r, text_mask):
            t=trim_horizontal_to_pane_evidence(crop, r, text_mask)
            if t is not None and t.w < 120:
                t=None
        elif 55 <= r.w <= 105 and horizontal_row_band_count(crop, r, text_mask) >= 3 and len(horizontal_pane_segments_from_ink(crop, r, text_mask)) >= 1:
            t=trim_horizontal_to_pane_evidence(crop, r, text_mask)
        if t is not None:
            if not any(abs(t.cx-o.cx)<35 and abs(t.cy-o.cy)<12 for o in keep if o.orientation=='horizontal'):
                keep.append(t)

    # Recover compact horizontal cap windows that have enough cap-band evidence
    # but were too short or measurement-line-interrupted for two-pane logic.
    for recovered in recover_compact_horizontal_cap_windows(crop, raw, keep, text_mask):
        if not _is_duplicate_candidate(recovered, keep):
            keep.append(recovered)

    # Remove any single-pane horizontal candidate mostly covered by a wider
    # same-row two-pane candidate. This removes the remaining wall-space/single
    # duplicate after a valid two-panel window has been formed.
    keep2=[]
    for c in keep:
        if c.orientation=='horizontal':
            covered=False
            for o in keep:
                if o is c or o.orientation!='horizontal':
                    continue
                if abs(o.cy-c.cy)>8 or o.w <= c.w*1.35:
                    continue
                overlap=max(0,min(c.x+c.w,o.x+o.w)-max(c.x,o.x))
                if overlap >= c.w*0.55:
                    covered=True
                    break
            if covered:
                continue
        keep2.append(c)
    return suppress_overlaps(keep2, tol=28)

def run_stantec(pdf:Path, pages:list[int], outdir:Path, dpi:int, group_homes:list[str]|None, save_raw:bool)->list[list[object]]:
    rows=[]
    for page in pages:
        page_img, zoom, page_rect=render_page_dpi(pdf,page,dpi)
        cv2.imwrite(str(outdir/f'page{page}_rendered.png'),page_img)
        titles,_=find_stantec_plan_titles(pdf,page)
        title_map={gh:bbox for gh,txt,bbox in titles}
        selected=group_homes or list(title_map.keys())
        locator=page_img.copy()
        for gh in selected:
            if gh not in title_map:
                print(f'WARNING: page {page}: Group Home {gh} title not found'); continue
            x0,y0,x1,y1=stantec_crop_from_title(title_map[gh],page_rect,zoom)
            crop=page_img[y0:y1,x0:x1].copy()
            base=f'page{page}_group_home_{gh}'
            text_mask=crop_text_mask(pdf, page, zoom, (x0,y0,x1,y1), crop.shape)
            filtered, raw=stantec_wall_band_windows(crop, save_debug_dir=(outdir/f'page{page}_group_home_{gh}_debug') if save_raw else None, text_mask=text_mask)
            # v14 deterministic thin-interior-wall cap pass. Add only non-overlapping
            # horizontal cap candidates missed by thick-wall band detection.
            interior_caps=detect_generic_interior_thin_horizontal_caps(crop, text_mask=text_mask)
            for cap in interior_caps:
                if not any(rect_iou(cap, f)>0.08 or (abs(cap.cx-f.cx)<36 and abs(cap.cy-f.cy)<28) for f in filtered):
                    filtered.append(cap)

            # v17 generic rescue: some real cap windows in thin/hatch-heavy wall conditions
            # are present in the raw wall-band candidates but get rejected by conservative
            # post-filters. Re-add only normal-sized vertical cap candidates, with native
            # PDF text suppression and duplicate suppression. This is still rule-based and
            # not room-specific.
            for r in raw:
                if r.orientation == 'vertical' and 10 <= r.w <= 55 and 50 <= r.h <= 125:
                    if text_mask is not None and density(text_mask, r.x-4, r.y-4, r.x+r.w+4, r.y+r.h+4) > 0.08:
                        continue
                    if not any(rect_iou(r, f)>0.08 or (abs(r.cx-f.cx)<30 and abs(r.cy-f.cy)<35) for f in filtered):
                        filtered.append(r)

            # Generic cap-ink trim: remove empty wall-space after/before the actual
            # window cap and reject sparse door/gap candidates.
            trimmed=[]
            for f in filtered:
                t=trim_candidate_to_cap_ink(crop, f, text_mask=text_mask)
                if t is not None:
                    trimmed.append(t)
            filtered=suppress_overlaps(trimmed, tol=28)
            filtered=reject_space_after_window_artifacts(crop, filtered, text_mask=text_mask)
            filtered=promote_upper_component_and_reject_wall_space(crop, filtered, raw, text_mask=text_mask)
            for rec in split_long_vertical_raw_to_upper_caps(crop, raw, filtered, text_mask=text_mask):
                filtered.append(rec)
            filtered=refine_horizontal_two_pane_from_raw(crop, filtered, raw, text_mask=text_mask)
            filtered=suppress_overlaps(filtered, tol=28)
            if save_raw:
                write_candidate_csv(outdir/f'{base}_interior_thin_wall_cap_candidates.csv',interior_caps,{'page':page,'group_home':gh})
                cv2.imwrite(str(outdir/f'{base}_interior_thin_wall_cap_candidates_annotated.png'),draw_boxes(crop,interior_caps,'I'))
            if save_raw:
                cv2.imwrite(str(outdir/f'{base}_text_suppression_mask.png'), text_mask)
            cv2.rectangle(locator,(x0,y0),(x1,y1),(0,0,255),5)
            cv2.putText(locator,f'GH{gh}',(x0,max(40,y0-15)),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,255),3,cv2.LINE_AA)
            cv2.imwrite(str(outdir/f'{base}_crop.png'),crop)
            cv2.imwrite(str(outdir/f'{base}_window_candidates_annotated.png'),draw_boxes(crop,filtered,'W'))
            write_candidate_csv(outdir/f'{base}_window_candidates.csv',filtered,{'page':page,'group_home':gh})
            if save_raw:
                cv2.imwrite(str(outdir/f'{base}_raw_wall_band_candidates_annotated.png'),draw_boxes(crop,raw,'R'))
                write_candidate_csv(outdir/f'{base}_raw_wall_band_candidates.csv',raw,{'page':page,'group_home':gh})
            rows.append([page,gh,len(raw),len(filtered),f'{base}_window_candidates_annotated.png',f'{base}_window_candidates.csv'])
        cv2.imwrite(str(outdir/f'page{page}_group_home_crop_locations.png'),locator)
    return rows


# ---------------- App adapter API ----------------

def _load_rendered_page_image(pdf:Path, page_num:int, page_image_path:Path|None, dpi:int)->tuple[np.ndarray,float,fitz.Rect]:
    page_rect=fitz.open(str(pdf))[page_num-1].rect
    if page_image_path and page_image_path.exists():
        page_img=cv2.imread(str(page_image_path), cv2.IMREAD_COLOR)
        if page_img is None:
            raise RuntimeError(f'Could not read rendered page image: {page_image_path}')
        zoom=page_img.shape[1] / page_rect.width
        return page_img, zoom, page_rect
    return render_page_dpi(pdf,page_num,dpi)


def _candidate_to_app_window(candidate:Candidate, offset_x:int=0, offset_y:int=0, idx:int=1, label_prefix:str='W')->dict:
    return {
        'label': f'{label_prefix}-{idx:02d}',
        'box_px': [
            int(candidate.y + offset_y),
            int(candidate.x + offset_x),
            int(candidate.y + candidate.h + offset_y),
            int(candidate.x + candidate.w + offset_x),
        ],
        'detector': 'deterministic-stantec',
        'orientation': candidate.orientation,
        'source': candidate.source,
        'score': round(float(candidate.score), 5),
        'exterior_side': candidate.exterior_side,
    }


def _run_stantec_crop(crop:np.ndarray, pdf:Path, page_num:int, zoom:float, crop_rect_px:tuple[int,int,int,int], save_debug_dir:Path|None=None)->list[Candidate]:
    text_mask=crop_text_mask(pdf, page_num, zoom, crop_rect_px, crop.shape)
    filtered, raw=stantec_wall_band_windows(crop, save_debug_dir=save_debug_dir, text_mask=text_mask)

    interior_caps=detect_generic_interior_thin_horizontal_caps(crop, text_mask=text_mask)
    for cap in interior_caps:
        if not any(rect_iou(cap, f)>0.08 or (abs(cap.cx-f.cx)<36 and abs(cap.cy-f.cy)<28) for f in filtered):
            filtered.append(cap)

    for r in raw:
        if r.orientation == 'vertical' and 10 <= r.w <= 55 and 50 <= r.h <= 125:
            if text_mask is not None and density(text_mask, r.x-4, r.y-4, r.x+r.w+4, r.y+r.h+4) > 0.08:
                continue
            if not any(rect_iou(r, f)>0.08 or (abs(r.cx-f.cx)<30 and abs(r.cy-f.cy)<35) for f in filtered):
                filtered.append(r)

    trimmed=[]
    for f in filtered:
        t=trim_candidate_to_cap_ink(crop, f, text_mask=text_mask)
        if t is not None:
            trimmed.append(t)
    filtered=suppress_overlaps(trimmed, tol=28)
    filtered=reject_space_after_window_artifacts(crop, filtered, text_mask=text_mask)
    filtered=promote_upper_component_and_reject_wall_space(crop, filtered, raw, text_mask=text_mask)
    for rec in split_long_vertical_raw_to_upper_caps(crop, raw, filtered, text_mask=text_mask):
        filtered.append(rec)
    filtered=refine_horizontal_two_pane_from_raw(crop, filtered, raw, text_mask=text_mask)
    return suppress_overlaps(filtered, tol=28)


def detect_stantec_plan_regions(pdf_path:str|Path, page_num:int, page_image_path:str|Path|None=None, dpi:int=200)->list[dict]:
    """
    Returns app-style plan regions using Stantec/AB group-home title blocks.
    Falls back to an empty list when the PDF page does not match this profile.
    """
    pdf=Path(pdf_path)
    img_path=Path(page_image_path) if page_image_path else None
    page_img, zoom, page_rect=_load_rendered_page_image(pdf,page_num,img_path,dpi)
    titles,_=find_stantec_plan_titles(pdf,page_num)
    regions=[]
    for gh,txt,bbox in titles:
        x0,y0,x1,y1=stantec_crop_from_title(bbox,page_rect,zoom)
        x0=max(0,min(page_img.shape[1],x0)); x1=max(0,min(page_img.shape[1],x1))
        y0=max(0,min(page_img.shape[0],y0)); y1=max(0,min(page_img.shape[0],y1))
        if x1<=x0 or y1<=y0:
            continue
        regions.append({
            'label': f'Group Home {gh}',
            'group_home': str(gh),
            'box_px': [int(y0), int(x0), int(y1), int(x1)],
            'width': int(x1-x0),
            'height': int(y1-y0),
            'source': 'deterministic-stantec-title',
        })
    return sorted(regions, key=lambda r:(r['box_px'][0], r['box_px'][1]))


def is_stantec_pdf(pdf_path:str|Path, pages:list[int]|None=None)->bool:
    try:
        profile,_reason=auto_select_profile(Path(pdf_path),pages)
    except SystemExit:
        return False
    return profile == 'stantec_group_homes'


def detect_stantec_windows_in_region(
    pdf_path:str|Path,
    page_num:int,
    page_image_path:str|Path,
    region:list[int],
    dpi:int=200,
    save_debug_dir:str|Path|None=None,
)->list[dict]:
    """
    Runs the merged Stantec detector against one app-selected crop and returns
    full-page app windows: {'label': str, 'box_px': [ymin,xmin,ymax,xmax]}.
    """
    if len(region) != 4:
        raise ValueError('region must be [ymin, xmin, ymax, xmax].')

    pdf=Path(pdf_path)
    img_path=Path(page_image_path)
    page_img, zoom, _page_rect=_load_rendered_page_image(pdf,page_num,img_path,dpi)
    height,width=page_img.shape[:2]
    ymin,xmin,ymax,xmax=[int(v) for v in region]
    ymin=max(0,min(height,ymin)); ymax=max(0,min(height,ymax))
    xmin=max(0,min(width,xmin)); xmax=max(0,min(width,xmax))
    if xmin>=xmax or ymin>=ymax:
        raise ValueError('Invalid region: zero or negative area.')

    crop=page_img[ymin:ymax,xmin:xmax].copy()
    debug_path=Path(save_debug_dir) if save_debug_dir else None
    candidates=_run_stantec_crop(crop,pdf,page_num,zoom,(xmin,ymin,xmax,ymax),debug_path)
    return [_candidate_to_app_window(c, offset_x=xmin, offset_y=ymin, idx=i) for i,c in enumerate(candidates,1)]


def detect_stantec_windows_for_page(
    pdf_path:str|Path,
    page_num:int,
    page_image_path:str|Path|None=None,
    dpi:int=200,
    group_homes:list[str]|None=None,
    save_debug_dir:str|Path|None=None,
)->list[dict]:
    """
    Detects windows for all requested Stantec group-home title crops on a page.
    Coordinates are translated into full rendered-page pixel space.
    """
    pdf=Path(pdf_path)
    img_path=Path(page_image_path) if page_image_path else None
    page_img, zoom, page_rect=_load_rendered_page_image(pdf,page_num,img_path,dpi)
    titles,_=find_stantec_plan_titles(pdf,page_num)
    title_map={gh:bbox for gh,txt,bbox in titles}
    selected=group_homes or list(title_map.keys())
    all_windows=[]
    for gh in selected:
        if gh not in title_map:
            continue
        x0,y0,x1,y1=stantec_crop_from_title(title_map[gh],page_rect,zoom)
        x0=max(0,min(page_img.shape[1],x0)); x1=max(0,min(page_img.shape[1],x1))
        y0=max(0,min(page_img.shape[0],y0)); y1=max(0,min(page_img.shape[0],y1))
        if x1<=x0 or y1<=y0:
            continue
        crop=page_img[y0:y1,x0:x1].copy()
        debug_path=Path(save_debug_dir)/f'page{page_num}_group_home_{gh}_debug' if save_debug_dir else None
        candidates=_run_stantec_crop(crop,pdf,page_num,zoom,(x0,y0,x1,y1),debug_path)
        for idx,candidate in enumerate(candidates,1):
            window=_candidate_to_app_window(candidate, offset_x=x0, offset_y=y0, idx=idx, label_prefix=f'GH{gh}-W')
            window['group_home']=str(gh)
            all_windows.append(window)
    return sorted(all_windows, key=lambda w:(w['box_px'][0], w['box_px'][1]))


def write_summary(outdir:Path, rows:list[list[object]]):
    with (outdir/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['page','group_home','raw_candidate_count','filtered_candidate_count','annotated_png','csv']); w.writerows(rows)


def zip_dir(outdir:Path, zip_path:Path):
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in outdir.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(outdir.parent))


def main():
    ap=argparse.ArgumentParser(description='Deterministic wall-band + interior thin-wall cap detector v15, no per-window coordinate overrides')
    ap.add_argument('pdf',type=Path)
    ap.add_argument('--profile',choices=['auto','stantec_group_homes'],default='auto')
    ap.add_argument('--pages',default='18')
    ap.add_argument('--group-homes',nargs='*',default=None)
    ap.add_argument('--outdir',type=Path,default=Path('wall_band_window_output'))
    ap.add_argument('--dpi',type=int,default=200)
    ap.add_argument('--save-raw',action='store_true')
    ap.add_argument('--zip',action='store_true')
    args=ap.parse_args()
    doc=fitz.open(str(args.pdf)); pages=parse_pages(args.pages,len(doc))
    profile=args.profile; reason='manual override'
    if profile=='auto': profile,reason=auto_select_profile(args.pdf,pages)
    if profile!='stantec_group_homes': raise SystemExit('v14 currently runs the Stantec wall-band profile only.')
    args.outdir.mkdir(parents=True,exist_ok=True)
    print(f'Profile: {profile} ({reason})')
    rows=run_stantec(args.pdf,pages,args.outdir,args.dpi,args.group_homes,args.save_raw)
    write_summary(args.outdir,rows)
    for page,gh,raw,filtered,ann,csv_name in rows:
        print(f'Page {page} / Group Home {gh}: {filtered} filtered candidates from {raw} raw wall-band candidates')
    if args.zip:
        zip_path=args.outdir.with_suffix('.zip'); zip_dir(args.outdir,zip_path); print(f'ZIP: {zip_path}')

if __name__=='__main__': main()
