import os
import shutil
import subprocess



def convert_files(input_dir, output_dir, codec: str, bitrate: str):
    class SkipDir(Exception): pass
    # Startup
    print(f'Preparing files for {codec} at bitrate {bitrate}')
    # Setup encoder
    if codec == "ogg":
        encoder = "libopus"
    else:
        raise Exception("Codec not supported yet")

    # Recursively iterate through all files and directories
    for root, dirs, files in os.walk(input_dir):
        
        # Block the loop for irrelevant folders
        block = ["Code", "Evaluation","zip_files"]
        try:
            for i in block:
                if i in root.split("/"):
                    raise SkipDir
        except SkipDir:
            continue

        # Remove the files that are not needed
        try:
            dirs.remove("Code")
            dirs.remove("Evaluation")
            dirs.remove("zip_files")
        except ValueError:
            pass

        # Create corresponding directories in the output directory
        for dir in dirs:
            os.makedirs(os.path.join(output_dir, dir), exist_ok=True)
        
        # Convert WAV files to OGG and then back to WAV
        for file in files:
            print(file)
            if file.lower().endswith('.wav'):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, root.split("/")[-1], file)

                # Convert WAV to OGG
                '''
                ffmpeg -i input.wav -c:a libopus -b:a 5.5k -ar 8000 output_550.ogg
                ffmpeg -i output_550.ogg -ar 8000 output_550_ogg.wav
                '''
                encoded_file = os.path.splitext(output_file)[0] + f'.{codec}'
                subprocess.run(['ffmpeg', '-i', input_file, '-c:a', encoder, '-b:a', bitrate, encoded_file], check=True)

                # Convert OGG back to WAV
                subprocess.run(['ffmpeg', '-i', encoded_file, '-ar', '8000', output_file], check=True)

                # Remove OGG File
                subprocess.run(['rm', encoded_file], check=True)

if __name__ == '__main__':
    input_folder = '/scratch/jiaqi006/others/Yaseen_CHSSUMF/original'
    
    for bitrate in ['4.5k', '5.5k', '7.7k']:
        output_folder = f'/scratch/jiaqi006/others/Yaseen_CHSSUMF/ogg{bitrate}'
        convert_files(input_folder, output_folder, "ogg", bitrate)
