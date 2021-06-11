with open('tee.txt', 'rb') as source_file:
    with open('cleaned_logfile.txt', 'w+b') as dest_file:
        dest_file.write(source_file.read().decode('utf-16').encode('utf-8'))