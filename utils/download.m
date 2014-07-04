function download(from, to)
  if ~exist(to, 'file')
    fprintf('Downloading file %s to %s. It might take few minutes.', from, to);
    urlwrite(from, to);
    fprintf('File %s downloaded successfully.', from);
  else
    fprintf('File %s exists. Skipping downloading.', from);
  end
end