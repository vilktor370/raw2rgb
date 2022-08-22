import raw2rgb
def main():
    raw_cvtr = raw2rgb.convert.RawConvert()
    raw_cvtr.toRGB("data/raw/sample.tiff","data/rgb/sample.png")
    
if __name__ == "__main__":
    main()