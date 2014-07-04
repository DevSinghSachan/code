function download(from, to)
  if ~exist(to, 'file')
    fprintf('Downloading file %s to %s. It might take few minutes.\n', from, to);
    urlwrite(from, to);
    fprintf('File %s downloaded successfully.\n', from);
  else
    fprintf('File %s exists. Skipping downloading.\n', to);
  end
end