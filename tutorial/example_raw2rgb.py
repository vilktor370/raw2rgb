from raw2rgb import RawConvert
def main():
    raw_cvtr = RawConvert()
    raw_cvtr.toRGB("data/raw/sample.tiff","data/rgb/sample.png")
    
if __name__ == "__main__":
    main()