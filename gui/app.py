import json
import tkinter as tk
from pathlib import Path
import customtkinter as ctk
from PIL import Image
import fitz  # PyMuPDF

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class PatentApp(ctk.CTk):
    def __init__(self, output_base_dir: Path):
        super().__init__()
        self.title("Patent Cluster Viewer")
        self.geometry("1400x850")
        
        self.output_base_dir = output_base_dir
        self.mapping_dir = output_base_dir / "visualized" / "mapping"
        self.data_dir = Path("data")
        
        # 画像のガベージコレクション対策用リスト
        self.images = []
        
        # UIレイアウト構成
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0) # サイドバーは初期幅0(非表示)

        # 1. メイン画面 (マップ表示タブ)
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.tabview.add("脅威マップ")
        self.tabview.add("技術マップ")

        self.setup_map_tab("脅威マップ", "threat_map.png", "threat_data.json", base_width=800)
        self.setup_map_tab("技術マップ", "tech_map.png", "tech_data.json", base_width=1000)

        # 2. サイドバー (文献リスト)
        self.sidebar_frame = ctk.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid_propagate(False)
        
        self.close_btn = ctk.CTkButton(self.sidebar_frame, text="✖ 閉じる", width=100, 
                                       command=self.close_sidebar, fg_color="transparent", border_width=1)
        self.close_btn.pack(pady=10, padx=10, anchor="e")

        self.sidebar_title = ctk.CTkLabel(self.sidebar_frame, text="選択セルの文献一覧", font=("bold", 16))
        self.sidebar_title.pack(pady=5)

        self.list_frame = ctk.CTkScrollableFrame(self.sidebar_frame)
        self.list_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_map_tab(self, tab_name, img_name, json_name, base_width):
        tab = self.tabview.tab(tab_name)
        
        img_path = self.mapping_dir / img_name
        json_path = self.mapping_dir / json_name
        
        if not img_path.exists() or not json_path.exists():
            ctk.CTkLabel(tab, text=f"データが見つかりません:\n{img_name}").pack(pady=50)
            return
            
        with open(json_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            
        # 画像表示 (dpi=300の巨大画像を画面サイズに合わせて縮小)
        pil_img = Image.open(img_path)
        w_percent = base_width / float(pil_img.size[0])
        h_size = int(float(pil_img.size[1]) * w_percent)
        pil_img_resized = pil_img.resize((base_width, h_size), Image.Resampling.LANCZOS)
        
        ctk_img = ctk.CTkImage(light_image=pil_img_resized, dark_image=pil_img_resized, size=(base_width, h_size))
        self.images.append(ctk_img)  # 参照を保持
        
        # スクロールエリアを用意し、中央に画像を配置
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True)
        
        img_label = ctk.CTkLabel(scroll_frame, image=ctk_img, text="")
        img_label.pack(expand=True, pady=10)
        
        # クリックイベントのバインド (縮小後のサイズを渡す)
        img_label.bind("<Button-1>", lambda event: self.on_map_click(event, pil_img_resized.size, meta_data, tab_name))

    def on_map_click(self, event, img_size, meta_data, tab_name):
        w, h = img_size
        bbox = meta_data["bbox"]
        x_size = meta_data["x_size"]
        y_size = meta_data["y_size"]

        # 画像内の余白を除いたプロットエリア(Axes)のピクセル座標
        left, right = w * bbox["left"], w * bbox["right"]
        # Y軸はTkinterでは上が0、下がh
        top, bottom = h * (1 - bbox["top"]), h * (1 - bbox["bottom"])

        px, py = event.x, event.y

        # プロットエリア外のクリックは無視
        if not (left <= px <= right and top <= py <= bottom):
            return

        # グリッドのインデックスを計算
        col_idx = int((px - left) / (right - left) * x_size)
        
        if tab_name == "脅威マップ":
            # 脅威マップは下原点（0が一番下）
            row_idx = int((bottom - py) / (bottom - top) * y_size)
        else:
            # 技術マップは上原点（0が一番上）
            row_idx = int((py - top) / (bottom - top) * y_size)

        col_idx = min(col_idx, x_size - 1)
        row_idx = min(row_idx, y_size - 1)
        
        cell_key = f"{col_idx},{row_idx}"
        cell_docs = meta_data["cells"].get(cell_key, {})

        if cell_docs:
            self.show_sidebar(cell_docs)

    def show_sidebar(self, cell_docs):
        self.sidebar_frame.grid(row=0, column=1, sticky="nsew")
        
        # 既存のウィジェットをクリア
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        # Tree形式で一覧を生成
        for category, docs in cell_docs.items():
            cat_label = ctk.CTkLabel(self.list_frame, text=f"📂 {category}", font=("bold", 14), text_color="#1E90FF")
            cat_label.pack(anchor="w", pady=(10, 0))
            
            for i, doc in enumerate(docs):
                prefix = "└─ " if i == len(docs) - 1 else "├─ "
                doc_frame = ctk.CTkFrame(self.list_frame, fg_color="transparent")
                doc_frame.pack(fill="x", padx=15, pady=2)
                
                lbl_prefix = ctk.CTkLabel(doc_frame, text=prefix, text_color="gray")
                lbl_prefix.pack(side="left")
                
                # リンク風ボタン
                btn_doc = ctk.CTkButton(doc_frame, text=doc, anchor="w", fg_color="transparent", 
                                        text_color=("black", "white"), hover_color="#D3D3D3",
                                        command=lambda d=doc: self.open_pdf_preview(d))
                btn_doc.pack(side="left", fill="x", expand=True)

    def close_sidebar(self):
        self.sidebar_frame.grid_forget()

    def find_pdf_path(self, pdf_name):
        for pdf_path in self.data_dir.rglob("*.pdf"):
            if pdf_path.stem == pdf_name:
                return pdf_path
        return None

    def open_pdf_preview(self, pdf_name):
        pdf_path = self.find_pdf_path(pdf_name)
        if not pdf_path:
            tk.messagebox.showerror("エラー", f"PDFが見つかりません:\n{pdf_name}")
            return

        # モーダルウィンドウの作成
        preview_win = ctk.CTkToplevel(self)
        preview_win.title(f"PDF プレビュー - {pdf_name}")
        preview_win.geometry("900x900")
        preview_win.grab_set() # 他の操作をブロック

        top_frame = ctk.CTkFrame(preview_win, fg_color="transparent")
        top_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(top_frame, text=f"📄 {pdf_name}", font=("bold", 16)).pack(side="left")
        ctk.CTkButton(top_frame, text="✖ 閉じる", width=80, command=preview_win.destroy).pack(side="right")

        scroll_area = ctk.CTkScrollableFrame(preview_win)
        scroll_area.pack(fill="both", expand=True, padx=10, pady=10)

        try:
            doc = fitz.open(pdf_path)
            max_pages = min(3, len(doc))
            for i in range(max_pages):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                
                # サムネイル画像の参照を保持
                self.images.append(ctk_img)
                
                ctk.CTkLabel(scroll_area, text=f"- Page {i+1} -", text_color="gray").pack(pady=5)
                ctk.CTkLabel(scroll_area, image=ctk_img, text="").pack(pady=(0, 20))
                
        except Exception as e:
            ctk.CTkLabel(scroll_area, text=f"PDFの読み込みに失敗しました:\n{e}").pack(pady=20)

def run_gui(output_base_dir: Path):
    app = PatentApp(output_base_dir)
    app.mainloop()