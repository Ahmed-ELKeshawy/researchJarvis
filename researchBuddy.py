import tkinter as tk
import customtkinter as ctk
from tkinter import ttk, filedialog, scrolledtext,font
import os
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from langchain.embeddings import GPT4AllEmbeddings
from pdfminer.high_level import extract_text,extract_pages
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pdfminer.layout import LTTextContainer,LTChar
import fitz  # PyMuPDF
from pdfminer.layout import LAParams
import traceback
from langchain_community.vectorstores import Chroma
import threading


# Set CustomTkinter appearance
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class DocumentQAInterface:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Research Assistant")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Set window size to maximum screen size
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # Optional: Make it truly fullscreen (without window decorations)
        # self.root.attributes('-fullscreen', True)
        
        # Alternative: Maximize the window
        self.root.state('zoomed')  # For Windows
        # self.root.attributes('-zoomed', True)  # For Linux
        
        # Create upload folder
        self.upload_folder = 'uploaded_files'
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # Initialize components
        self.initialize_components()
        self.create_widgets()


    def initialize_components(self):
        """Initialize all required components for the QA system"""
        try:
            # Initialize status to track progress
            self.initialization_status = "Starting initialization...\n"
            
            # Initialize embeddings
            self.initialization_status += "Initializing embeddings...\n"
            self.embeddings = GPT4AllEmbeddings(client=any)
            
            # Initialize vector store with metadata filtering
            self.initialization_status += "Initializing vector store...\n"
            
            
            # Define metadata fields that should be preserved
            self.vectorstore = Chroma(
                persist_directory="TesterFinal",
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
                
            )
            
            # Initialize LLM
            self.initialization_status += "Initializing Ollama...\n"
            self.llm = Ollama(model="llama3.2:1b")
            
            # Initialize prompt template
            self.initialization_status += "Setting up prompt template...\n"
            self.prompt_template = """
            Context: {context}
            Question: {question}
            Answer the question based on the context provided. If you cannot find 
            the answer in the context, say "I cannot find the answer in the provided documents."
            Answer:
            """
            self.QA_PROMPT = PromptTemplate(
                template=self.prompt_template,
                input_variables=["context", "question"]
            )
            
            # Initialize QA chain
            self.initialization_status += "Setting up QA chain...\n"
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": self.QA_PROMPT},
                return_source_documents=True
            )
            
            self.initialization_status += "Initialization completed successfully!\n"
            
        except Exception as e:
            self.initialization_status = f"Error during initialization: {str(e)}\n"
            print(f"Initialization error: {str(e)}")

    def create_widgets(self):
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill="both", expand=True, padx=5, pady=5)

        # Header Section
        header_frame = ctk.CTkFrame(self.main_container)
        header_frame.pack(fill="x", pady=(0, 20))

        title = ctk.CTkLabel(
            header_frame,
            text="Research Assistant",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title.pack(pady=(10, 5))

        subtitle = ctk.CTkLabel(
            header_frame,
            text="Drop your research papers, ask anything, get instant answers!",
            font=ctk.CTkFont(size=16)
        )
        subtitle.pack(pady=(0, 10))

        # Create two-column layout using PanedWindow instead of Frame
        content_frame = ttk.PanedWindow(self.main_container, orient="horizontal")
        content_frame.pack(fill="both", expand=True)

        # Left Panel (Document Management)
        left_panel = ctk.CTkFrame(content_frame)
        
        # Right Panel (Query Interface)
        right_panel = ctk.CTkFrame(content_frame)
        
        # Add both panels to the PanedWindow
        content_frame.add(left_panel, weight=1)
        content_frame.add(right_panel, weight=1)

        # Document List Frame
        doc_list_frame = ctk.CTkFrame(left_panel)
        doc_list_frame.pack(fill="both", expand=True, pady=(0, 0))

        doc_list_label = ctk.CTkLabel(
            doc_list_frame,
            text="📁 Your Research Papers",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        doc_list_label.pack(pady=10)

        # Style the Treeview to match dark theme - Move this BEFORE creating the Treeview
        style = ttk.Style()
        
        # Fix for the white background issue
        style.layout("Treeview", [
            ('Treeview.treearea', {'sticky': 'nswe'})
        ])
        
        # Configure the base Treeview style
        style.configure("Treeview",
                        background="#2b2b2b",
                        foreground="white",
                        fieldbackground="#2b2b2b",
                        borderwidth=0,
                        rowheight=50)
        
        # Style the scrollbar to be minimal
        style.configure("Minimal.TScrollbar",
                       background="#2b2b2b",      # Background of scrollbar
                       darkcolor="#2b2b2b",
                       lightcolor="#2b2b2b",
                       troughcolor="#1f1f1f",     # Color of the trough/track
                       bordercolor="#2b2b2b",
                       borderwidth=0,
                       relief="flat",
                       width=8)                    # Make scrollbar thinner
        
        # Define the layout for the minimal scrollbar
        style.layout("Minimal.TScrollbar", [
            ("Minimal.TScrollbar.trough", {
                "children": [
                    ("Minimal.TScrollbar.thumb", {
                        "sticky": "nswe"
                    })
                ],
                "sticky": "ns"
            })
        ])
        
        # Configure scrollbar colors and hover effects
        style.map("Minimal.TScrollbar",
                 background=[('pressed', '#ff3333'),     # Bright red when pressed
                           ('active', '#cc0000'),        # Darker red when hovered
                           ('!active', '#990000')],      # Base red color
                 troughcolor=[('!disabled', '#1f1f1f')]) # Dark gray trough
        
        # Configure the Heading style explicitly
        style.element_create("Custom.Treeheading.border", "from", "default")
        style.layout("Custom.Treeview.Heading", [
            ("Custom.Treeheading.border", {'sticky': 'nswe', 'children': [
                ("Custom.Treeheading.padding", {'sticky': 'nswe', 'children': [
                    ("Custom.Treeheading.image", {'side': 'right', 'sticky': ''}),
                    ("Custom.Treeheading.text", {'sticky': 'we'})
                ]})
            ]})
        ])
        
        style.configure("Custom.Treeview.Heading",
                       background="#2b2b2b",
                       foreground="white",
                       borderwidth=0,
                       relief="flat")
        
        style.map("Custom.Treeview.Heading",
                 background=[('active', '#2b2b2b')],
                 foreground=[('active', 'white')])
        
        # Create a container for Treeview and scrollbar
        tree_container = ctk.CTkFrame(
            doc_list_frame,
            fg_color="transparent"
        )
        tree_container.pack(fill="both", expand=True)
        
        # Add Treeview and scrollbar to the container
        self.file_list = ttk.Treeview(
            tree_container,  # Changed parent to tree_container
            columns=('Path', 'Filename'),
            show='headings',
            height=8,
            style="Custom.Treeview"
        )
        self.file_list.heading('Path', text='File Location')
        self.file_list.heading('Filename', text='Filename')
        
        # Add minimal scrollbar for the Treeview
        scrollbar = ttk.Scrollbar(
            tree_container,  # Changed parent to tree_container
            orient="vertical", 
            command=self.file_list.yview,
            style="Minimal.TScrollbar"
        )
        
        # Pack Treeview and scrollbar in the container
        self.file_list.configure(yscrollcommand=scrollbar.set)
        self.file_list.pack(side="left", fill="both", expand=True, padx=10, pady=5)
        scrollbar.pack(side="right", fill="y", pady=5)
        
        # Now add buttons below the tree_container
        button_container = ctk.CTkFrame(
            doc_list_frame,
            fg_color="transparent"
        )
        button_container.pack(fill="x", pady=20)

        self.select_button = ctk.CTkButton(
            button_container,
            text="Pick Files 📎",
            command=self.select_files,
            font=ctk.CTkFont(size=14)
        )
        self.select_button.pack(side="left", padx=5, expand=True)

        self.upload_button = ctk.CTkButton(
            button_container,
            text="Upload ⬆️",
            command=self.upload_files,
            font=ctk.CTkFont(size=14)
        )
        self.upload_button.pack(side="left", padx=5, expand=True)

        # Upload Status also moves into doc_list_frame
        self.upload_result = ctk.CTkLabel(
            doc_list_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.upload_result.pack(pady=5)

        # Progress Bar and Log Message
        self.upload_progress = ctk.CTkProgressBar(
            doc_list_frame,
            orientation="horizontal",
            mode="determinate",
            width=200
        )
        self.upload_progress.set(0)  # Set initial value
        self.upload_progress.pack(pady=5)

        self.upload_log = ctk.CTkLabel(
            doc_list_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.upload_log.pack(pady=0)

        # Query Section
        query_label = ctk.CTkLabel(
            right_panel,
            text="❓ Ask Away!",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        query_label.pack(pady=10)

        self.query_entry = ctk.CTkEntry(
            right_panel,
            placeholder_text="Type your question here...",
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.query_entry.pack(fill="x", padx=10, pady=(0, 10))

        # Button container for query and highlight buttons
        query_buttons_container = ctk.CTkFrame(right_panel, fg_color="transparent")
        query_buttons_container.pack(pady=(0, 10))

        self.query_button = ctk.CTkButton(
            query_buttons_container,
            text="Get Answer 🔍",
            command=self.submit_query,
            font=ctk.CTkFont(size=14)
        )
        self.query_button.pack(side="left", padx=5)

        self.highlight_button = ctk.CTkButton(
            query_buttons_container,
            text="View Sources 📄",
            command=lambda: self.highlight_source_documents(self.current_sources) if hasattr(self, 'current_sources') else None,
            font=ctk.CTkFont(size=14)
        )
        self.highlight_button.pack(side="left", padx=5)

        # Result Area
        self.query_result = ctk.CTkTextbox(
            right_panel,
            wrap="word",
            height=300,
            font=ctk.CTkFont(size=14)
        )
        self.query_result.pack(fill="both", expand=True, padx=10, pady=5)

      

        # Visualization Section
        viz_frame = ctk.CTkFrame(self.main_container, height=100)  # Back to 100
        viz_frame.pack(fill="x", pady=(5, 5))
        viz_frame.pack_propagate(False)  # Prevent frame from shrinking

        # Create a vertical container for label and button
        controls_container = ctk.CTkFrame(viz_frame, fg_color="transparent")
        controls_container.pack(expand=True, pady=2)  # Center vertically

        viz_label = ctk.CTkLabel(
            controls_container,
            text="🎨 Document Map",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        viz_label.pack(pady=(0, 15))  # Add padding between label and button

        self.update_viz_button = ctk.CTkButton(
            controls_container,
            text="Update Map 🔄",
            command=self.update_visualization,
            font=ctk.CTkFont(size=14)
        )
        self.update_viz_button.pack()

        # Removed plot_frame since it's not needed
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select PDF files",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        print(f"Selected files: {files}")  # Debug print
        
        # Clear existing items
        for item in self.file_list.get_children():
            self.file_list.delete(item)
            
        # Add new files to the list
        for file_path in files:
            filename = os.path.basename(file_path)
            print(f"Adding file: {file_path}, {filename}")  # Debug print
            self.file_list.insert('', tk.END, values=(file_path, filename))
        
        print(f"Total files in list: {len(self.file_list.get_children())}")  # Debug print
    
    def upload_files(self):
        # Start a new thread for file processing
        threading.Thread(target=self.process_files).start()

    def process_files(self):
        files_to_process = list(self.file_list.get_children())
        
        if not files_to_process:
            self.upload_result.configure(
                text="No files selected for upload.",
                text_color="red"
            )
            return

        successful_uploads = 0
        failed_uploads = 0
        total_files = len(files_to_process)

        # Initialize progress bar
        self.upload_progress.set(0)
        self.upload_progress.start()

        for index, item in enumerate(files_to_process):
            file_values = self.file_list.item(item)['values']
            
            if not file_values or len(file_values) < 2:
                self.upload_result.configure(
                    text="Invalid file selection. Please reselect files.",
                    text_color="red"
                )
                return

            file_path = file_values[0]
            filename = file_values[1]
            
            try:
                # Process the file and track success
                if self.load_pdf(file_path):
                    successful_uploads += 1
                    self.upload_log.configure(text=f"Uploaded: {filename}", text_color="green")
                else:
                    failed_uploads += 1
                    self.upload_log.configure(text=f"Failed to upload: {filename}", text_color="red")
            except Exception as e:
                failed_uploads += 1
                self.upload_log.configure(text=f"Error uploading {filename}: {e}", text_color="red")
                print(f"Unexpected error uploading {filename}: {e}")

            # Update progress bar (value between 0 and 1)
            self.upload_progress.set(float(index + 1) / total_files)

        # Stop progress bar
        self.upload_progress.stop()

        # Update result message using CTkLabel configuration
        if successful_uploads > 0 and failed_uploads == 0:
            self.upload_result.configure(
                text=f"Successfully uploaded {successful_uploads} file(s)!",
                text_color="green"
            )
        elif successful_uploads > 0 and failed_uploads > 0:
            self.upload_result.configure(
                text=f"Uploaded {successful_uploads} file(s), {failed_uploads} failed.",
                text_color="orange"
            )
        else:
            self.upload_result.configure(
                text="No files could be uploaded.",
                text_color="red"
            )
        
        # After successful upload, update visualization
        if successful_uploads > 0:
            self.update_visualization()
    
    def submit_query(self):
        query = self.query_entry.get()
        if query:
            try:
                # Run QA chain
                result = self.qa_chain({"query": query})
                
                # Store source documents for later use
                self.current_sources = result.get("source_documents", [])
                
                # Get the answer
                answer = result['result']
                
                # Prepare response text
                unique_file_names = {doc.metadata.get("fileName", "Unknown File") 
                                    for doc in self.current_sources}
                
                response_text = f"Answer:\n{answer}\n\n"
                response_text += "Context retrieved from the following document(s):\n"
                for file_name in unique_file_names:
                    response_text += f"- {file_name}\n"
                
                # Update query result
                self.query_result.delete(1.0, tk.END)
                self.query_result.insert(tk.END, response_text)
                
            except Exception as e:
                error_message = f"Error processing query: {str(e)}"
                self.query_result.delete(1.0, tk.END)
                self.query_result.insert(tk.END, error_message)
                print(error_message)
    
    def load_pdf(self, file_path):
        try:
            # Extensive text extraction attempts
            print("Attempting text extraction methods:")
            
            # Method 1: Standard extract_text
            try:
                fileText = extract_text(file_path)
                print(f"Standard extract_text - Length: {len(fileText)}")
                print(f"First 200 characters: {fileText[:200]}")
            except Exception as e:
                print(f"Standard extract_text failed: {e}")
                fileText = ""
            
            # Method 2: Extract text with custom LAParams
            if not fileText:
                try:
                    fileText = extract_text(file_path, laparams=LAParams(
                        line_margin=0.5, 
                        char_margin=2.0, 
                        boxes_flow=0.5
                    ))
                    print(f"Custom LAParams extract_text - Length: {len(fileText)}")
                except Exception as e:
                    print(f"Custom LAParams extraction failed: {e}")
            
            # Method 3: Manual page extraction
            if not fileText:
                try:
                    all_page_texts = []
                    for page_layout in extract_pages(file_path):
                        page_text = ""
                        for element in page_layout:
                            if hasattr(element, 'get_text'):
                                page_text += element.get_text()
                        all_page_texts.append(page_text)
                    
                    fileText = "\n".join(all_page_texts)
                    print(f"Manual page extraction - Length: {len(fileText)}")
                except Exception as e:
                    print(f"Manual page extraction failed: {e}")
                    traceback.print_exc()
            
            # Verify text extraction
            if not fileText or fileText.strip() == "":
                print("CRITICAL: No text could be extracted from the PDF")
                self.upload_result.config(
                    text=f"Could not extract text from {os.path.basename(file_path)}. Ensure it's a text-based PDF.",
                    foreground="red"
                )
                return False

            filename = os.path.basename(file_path)
            documentFile = self.split_text(fileText, filename, file_path)
            
            # Check if documentFile is empty
            if not documentFile:
                print(f"No valid chunks found in file {file_path}")
                self.upload_result.config(
                    text=f"No valid text chunks in {filename}. PDF might be scanned or image-based.",
                    foreground="red"
                )
                return False

            print(f"Added {len(documentFile)} chunks for file: {documentFile[0].metadata['Path']}")
            self.addDocToVec(documentFile)
            return True

        except Exception as e:
            print(f"Unexpected error processing {file_path}: {str(e)}")
            traceback.print_exc()
            self.upload_result.config(
                text=f"Unexpected error processing {os.path.basename(file_path)}: {str(e)}",
                foreground="red"
            )
            return False
    

    def addDocToVec(self, data):
        try:
            print(f"Attempting to add {len(data)} documents to vector store")
            
            # Debug print first document's metadata
            if data:
                print(f"First document metadata before adding to vector store: {data[0].metadata}")
            
            # Verify documents and their metadata
            valid_docs = []
            for doc in data:
                if doc.page_content.strip():
                    # Create new document with verified metadata
                    valid_docs.append(Document(
                        page_content=doc.page_content,
                        metadata={
                            "Path": doc.metadata.get("Path", ""),
                            "fileName": doc.metadata.get("fileName", ""),
                            "page": doc.metadata.get("page", 1),
                            "coords": doc.metadata.get("coords", "")
                        }
                    ))
                    print(f"Added document with coords: {doc.metadata.get('coords', '')[:100]}")
            
            print(f"Number of valid documents: {len(valid_docs)}")
            
            if not valid_docs:
                print("No valid documents to add to vector store")
                return
            
            # Add documents directly
            self.vectorstore.add_documents(valid_docs)
            
            # Verify metadata was saved
            if valid_docs:
                print("Verifying first document metadata after storage:")
                results = self.vectorstore.similarity_search(valid_docs[0].page_content, k=1)
                if results:
                    print(f"Retrieved metadata: {results[0].metadata}")
            
            self.vectorstore.persist()
            print('Successfully added files to vector store')
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            traceback.print_exc()

    def split_text(self, text, filename, path, chunk_size=1000, chunk_overlap=100):
        try:
            print("Starting text extraction with coordinates...")
            chunks_with_coords = []
            
            for page_num, page_layout in enumerate(extract_pages(path), start=1):
                print(f"Processing page {page_num}")
                page_text = ""
                page_coords = []
                
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text_content = element.get_text()
                        bbox = element.bbox
                        
                        # Debug print for coordinates
                        print(f"Found text element: {text_content[:50]}... at coords: {bbox}")
                        
                        page_text += text_content
                        page_coords.append(bbox)
                        
                        if len(page_text) >= chunk_size:
                            coords_str = ";".join([f"{x0},{y0},{x1},{y1}" 
                                                for x0,y0,x1,y1 in page_coords])
                            
                            # Verify coordinates are being stored
                            print(f"Creating chunk with coordinates: {coords_str[:100]}")
                            
                            document = Document(
                                page_content=page_text,
                                metadata={
                                    "Path": path,
                                    "fileName": filename,
                                    "page": page_num,
                                    "coords": coords_str
                                }
                            )
                            chunks_with_coords.append(document)
                            
                            # Keep overlap for next chunk
                            words = page_text.split()
                            page_text = " ".join(words[-50:])
                            page_coords = [page_coords[-1]]
                
                # Process remaining text on the page
                if page_text.strip():
                    coords_str = ";".join([f"{x0},{y0},{x1},{y1}" 
                                        for x0,y0,x1,y1 in page_coords])
                    
                    document = Document(
                        page_content=page_text,
                        metadata={
                            "Path": path,
                            "fileName": filename,
                            "page": page_num,
                            "coords": coords_str
                        }
                    )
                    chunks_with_coords.append(document)
            
            # Verify final chunks
            print(f"\nCreated {len(chunks_with_coords)} chunks with coordinates")
            if chunks_with_coords:
                print("Sample from first chunk:")
                print(f"Content: {chunks_with_coords[0].page_content[:100]}")
                print(f"Metadata: {chunks_with_coords[0].metadata}")
            
            return chunks_with_coords

        except Exception as e:
            print(f"Error in split_text: {e}")
            traceback.print_exc()
            return []

    def highlight_source_documents(self, source_documents):
        try:
            # Group documents by file path
            documents_by_file = {}
            for doc in source_documents:
                file_path = doc.metadata.get('Path')
                if file_path:
                    if file_path not in documents_by_file:
                        documents_by_file[file_path] = []
                    documents_by_file[file_path].append(doc)
            
            # Process each file separately
            for file_path, docs in documents_by_file.items():
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue
                
                try:
                    pdf_document = fitz.open(file_path)
                    highlights_added = False
                    
                    # Process all chunks from this file
                    for doc in docs:
                        coords_str = doc.metadata.get('coords')
                        page_num = doc.metadata.get('page', 1)
                        
                        if not coords_str:
                            print(f"No coordinates found for chunk in {file_path}")
                            continue
                        
                        try:
                            page = pdf_document[page_num - 1]  # Convert to 0-based index
                            
                            # Parse coordinates string
                            coord_sets = coords_str.split(";")
                            print(f"Processing {len(coord_sets)} coordinate sets on page {page_num}")
                            
                            for coord_set in coord_sets:
                                if not coord_set.strip():
                                    continue
                                
                                try:
                                    x0, y0, x1, y1 = map(float, coord_set.split(","))
                                    
                                    # Convert coordinates from PDFMiner to PyMuPDF
                                    page_height = page.rect.height
                                    rect = fitz.Rect(
                                        x0,                    # left
                                        page_height - y1,      # top
                                        x1,                    # right
                                        page_height - y0       # bottom
                                    )
                                    
                                    # Add highlight with some padding
                                    padding = 2
                                    rect.x0 -= padding
                                    rect.x1 += padding
                                    rect.y0 -= padding
                                    rect.y1 += padding
                                    
                                    # Make highlight more visible
                                    annot = page.add_highlight_annot(rect)
                                    annot.set_colors(stroke=(1, 1, 0))  # Bright yellow
                                    annot.set_opacity(0.5)
                                    annot.update()
                                    
                                    highlights_added = True
                                    print(f"Added highlight at {rect} on page {page_num}")
                                    
                                except Exception as coord_error:
                                    print(f"Error processing coordinate set {coord_set}: {coord_error}")
                                    continue
                                    
                        except Exception as page_error:
                            print(f"Error processing page {page_num}: {page_error}")
                            continue
                    
                    # Save and open the PDF only if highlights were added
                    if highlights_added:
                        base_path = os.path.splitext(file_path)[0]
                        highlighted_path = f"{base_path}_highlighted.pdf"
                        counter = 1
                        while os.path.exists(highlighted_path):
                            highlighted_path = f"{base_path}_highlighted_{counter}.pdf"
                            counter += 1
                        
                        pdf_document.save(highlighted_path)
                        print(f"Saved highlighted PDF: {highlighted_path}")
                        
                        # Open the highlighted PDF
                        import platform
                        if platform.system() == 'Darwin':       # macOS
                            os.system(f'open "{highlighted_path}"')
                        elif platform.system() == 'Windows':    # Windows
                            os.system(f'start "" "{highlighted_path}"')
                        else:                                   # Linux
                            os.system(f'xdg-open "{highlighted_path}"')
                    
                    pdf_document.close()
                    
                except Exception as e:
                    print(f"Error processing PDF {file_path}: {e}")
                    traceback.print_exc()
                    continue

        except Exception as e:
            print(f"Error in highlight_source_documents: {e}")
            traceback.print_exc()
    
    def visualize_embeddings(self):
        try:
            # Get all documents and their embeddings from Chroma
            collection = self.vectorstore._collection
            if collection.count() == 0:
                print("No documents found in the vector store")
                return
            
            # Get embeddings and metadata
            result = collection.get(include=['embeddings', 'metadatas', 'documents'])
            
            # Convert embeddings to numpy array
            embeddings = np.array(result['embeddings']) if result['embeddings'] else np.array([])
            if len(embeddings) == 0:
                print("No embeddings found in the vector store")
                return
            
            print(f"Found {len(embeddings)} documents with embeddings")
            print(f"Embedding shape: {embeddings.shape}")
            
            metadatas = result['metadatas']
            documents = result['documents']
            
            # Get unique documents and assign colors
            unique_docs = list(set(meta['fileName'] for meta in metadatas))
            colors = [f'rgb{tuple(int(x*255) for x in plt.cm.tab10(i)[:3])}' 
                     for i in range(len(unique_docs))]
            color_map = dict(zip(unique_docs, colors))
            
            # Reduce dimensionality with UMAP
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Create figure
            fig = go.Figure()
            
            # Plot points for each document
            for doc_name in unique_docs:
                # Get indices for this document
                doc_indices = [i for i, meta in enumerate(metadatas) 
                             if meta['fileName'] == doc_name]
                
                # Create hover text
                hover_texts = []
                for idx in doc_indices:
                    content_preview = documents[idx][:100] + "..." if len(documents[idx]) > 100 else documents[idx]
                    hover_texts.append(
                        f"File: {doc_name}<br>"
                        f"Chunk: {idx + 1}/{len(embeddings)}<br>"
                        f"Page: {metadatas[idx].get('page', 'N/A')}<br>"
                        f"Preview: {content_preview}"
                    )
                
                # Add scatter plot for this document
                fig.add_trace(
                    go.Scatter(
                        x=embeddings_2d[doc_indices, 0],
                        y=embeddings_2d[doc_indices, 1],
                        mode='markers',
                        name=doc_name,
                        marker=dict(
                            color=color_map[doc_name],
                            size=10,
                            opacity=0.6
                        ),
                        text=hover_texts,
                        hoverinfo='text',
                    )
                )
            
            # Update layout
            fig.update_layout(
                title='Document Embeddings Visualization',
                title_x=0.5,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                ),
                hovermode='closest',
                margin=dict(l=20, r=200, t=40, b=20),
                plot_bgcolor='white'
            )
            
            # Add grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
            
            return fig

        except Exception as e:
            print(f"Error creating visualization: {e}")
            traceback.print_exc()
            return None
    
    def update_visualization(self):
        try:
            # Clear existing plot
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Create new visualization
            fig = self.visualize_embeddings()
            if fig:
                # Create HTML string
                html = '<html><body>'
                html += fig.to_html(full_html=False, include_plotlyjs='cdn')
                html += '</body></html>'
                
                # Create a webview widget
                import webview
                browser_frame = webview.create_window('Embeddings Visualization', 
                                                    html=html,
                                                    width=800, 
                                                    height=600)
                webview.start()
                
        except Exception as e:
            print(f"Error updating visualization: {e}")
            traceback.print_exc()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DocumentQAInterface()
    app.run()