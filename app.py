import streamlit as st
import tempfile
import os
from model import resumemain
import zipfile
import base64

st.title('Resume Ranking Application')

st.markdown("""
    <style>
    .description-box {
        width: 80%;
        padding: 10px;
        /* background-color: #24292e;  GitHub dark grey */
        border-radius: 10px;
        border: 1px solid #ddd;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        transition: all 0.3s ease-in-out;
        color: #4CAF50; /* Green text */
    }

    .description-box:hover {
        white-space: normal;
        overflow: visible;
    }
    </style>

<div>Simply upload a ZIP file with multiple resumes, optionally add a job description file, and let the app analyze and rank the resumes for you. 🚀    </div>
""", unsafe_allow_html=True)

with st.container():
    st.subheader("Upload Files")

    uploaded_file = st.file_uploader(
        "Upload a zip file containing resume files",
        type=['zip'],
        key="resume_uploader"
    )

    job_description_file = st.file_uploader(
        "Upload a job description text file (optional)",
        type=['txt'],
        key="jd_uploader"
    )

if uploaded_file:
    st.success(f"✅ Resume file uploaded: {uploaded_file.name}")

if job_description_file:
    st.success(f"✅ Job description uploaded: {job_description_file.name}")

output_folder = "output"

if st.button("Process Resumes"):
    if uploaded_file is not None:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Path to save the uploaded zip file
            zip_path = os.path.join(temp_dir, uploaded_file.name)
            with open(zip_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Save job description if provided
            job_description_path = None
            if job_description_file:
                job_description_path = os.path.join(temp_dir, job_description_file.name)
                with open(job_description_path, 'wb') as f:
                    f.write(job_description_file.getvalue())

            # Process the zip file using the function from model.py
            with st.spinner('Processing resumes...'):
                results = resumemain(zip_path, job_description_path)

            # Check if results are available
            if results is not None and not results.empty:
                st.success('Resumes processed successfully!')
                st.write('### Results:')
                st.dataframe(results)

                # Provide option to download the results as CSV
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name='resume_results.csv',
                    mime='text/csv',
                )

                # Extract top 3 resume IDs
                # top_3_ids = results['ID'].head(3).tolist()
                top_3_ids = [str(i) for i in results['id'].head(3).tolist()]

                extracted_pdfs = {}
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                    for file_name in zip_ref.namelist():
                        base_name = os.path.splitext(os.path.basename(file_name))[0]
                        file_id = base_name.replace("candidate_", "")  # keep as-is
                        if file_id in top_3_ids and file_name.lower().endswith('.pdf'):
                            extracted_pdfs[file_id] = os.path.join(temp_dir, file_name)

                
                # Sort the extracted PDFs based on top_3_ids (so the highest-ranked comes first)
                sorted_pdfs = sorted(extracted_pdfs.items(), key=lambda x: top_3_ids.index(x[0]))

                # Display download buttons for top 3 resumes
                st.write("### Top 3 Resumes (PDFs)")
                if sorted_pdfs:
                    for resume_id, pdf_path in sorted_pdfs:
                        with open(pdf_path, "rb") as pdf_file:
                            pdf_data = pdf_file.read()
                            st.download_button(
                                label=f"Download Resume {resume_id}",
                                data=pdf_data,
                                file_name=f"candidate_{resume_id}.pdf",
                                mime="application/pdf"
                            )

                            # Embed PDF Viewer
                            base64_pdf = base64.b64encode(pdf_data).decode()
                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500"></iframe>'
                            st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    st.warning("No top resumes found in the zip file.")

            else:
                st.error("No valid results from the resume processing.")
    else:
        st.write('Please upload a zip file containing resume files.')
