class Configs:
    frame_length = 400
    frame_step = 160
    fft_length = 512
    batch_size = 16
    train_epochs = 1
    learning_rate = 1e-3
    model_path = "/kaggle/working/model"
    train_workers = 4
    vocab = [
        '،', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش',
        'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ـ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', 'ً', 'ٌ', 'ٍ',
        'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٓ', 'ٔ', 'ٕ', '٠', '١', '٢', '٣', '٤', '٥', '٦', '٧', '٨', '٩', '٪', '٫', '٬',
        '٭', 'ٰ', 'ٱ', 'ٹ', 'پ', 'چ', 'ڈ', 'ڑ', 'ژ', 'ک', 'گ', 'ں', 'ھ', 'ہ', 'ۂ', 'ۃ', 'ۆ', 'ۇ', 'ۈ', 'ۋ', 'ی',
        'ے', '۔', '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۾', 'ۿ', 'ﷺ'
    ]
    input_shape = [None, 257]

    def save(self):
        import json
        os.makedirs(self.model_path, exist_ok=True)
        with open(f"{self.model_path}/config.json", "w") as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        import json
        with open(path, "r") as f:
            data = json.load(f)
        config = cls()
        config.__dict__.update(data)
        return config
