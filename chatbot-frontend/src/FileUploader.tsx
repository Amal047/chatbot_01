import React, { useState } from "react";

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileSelect }) => {
  const [showUpload, setShowUpload] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
      setShowUpload(false);
    }
  };

  return (
    <div className="file-uploader">
      <button onClick={() => setShowUpload(!showUpload)}>+</button>
      {showUpload && <input type="file" onChange={handleFileChange} />}
    </div>
  );
};

export default FileUploader;
