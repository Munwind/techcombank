from flask import Flask, request, jsonify
import os
import re
import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
import PyPDF2
import pdfplumber
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import joblib
import warnings
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import torch

# ===== PDF PROCESSING FUNCTIONS =====
def read_pdf_with_pdfplumber(file_path: str) -> str:
    """Read PDF using pdfplumber - better for complex layouts"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {file_path} with pdfplumber: {e}")
    return text

def pdf_to_text(pdf_path: str) -> str:
    """Convert PDF to text"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    print(f'get pdf file {pdf_path}')
    text = read_pdf_with_pdfplumber(pdf_path)
    
    if not text.strip():
        raise ValueError("No text extracted from PDF")
    
    return text

esg_keywords = {
    'Environmental': {
        'climate_action': [
            # Climate & Carbon
            'khí hậu', 'biến đổi khí hậu', 'carbon', 'khí thải', 'phát thải', 'giảm phát thải',
            'trung hòa carbon', 'net zero', 'carbon footprint', 'tác động khí hậu', 'thích ứng khí hậu',
            'cam kết khí hậu', 'mục tiêu khí hậu', 'kế hoạch khí hậu', 'chiến lược khí hậu',
            'climate change', 'greenhouse gas', 'GHG', 'carbon emission', 'carbon neutral',
            'carbon reduction', 'decarbonization', 'climate resilience', 'climate action',
            'climate strategy', 'climate commitment', 'emission reduction', 'low carbon'
        ],
        
        'energy_transition': [
            # Energy & Technology
            'năng lượng tái tạo', 'năng lượng xanh', 'năng lượng sạch', 'tiết kiệm năng lượng',
            'hiệu quả năng lượng', 'chuyển đổi năng lượng', 'công nghệ xanh', 'công nghệ sạch',
            'điện mặt trời', 'điện gió', 'điện biomass', 'điện hạt nhân', 'thủy điện',
            'hệ thống năng lượng thông minh', 'lưới điện thông minh', 'lưu trữ năng lượng',
            'renewable energy', 'clean energy', 'green energy', 'energy efficiency',
            'energy transition', 'solar power', 'wind power', 'hydroelectric', 'biomass',
            'smart grid', 'energy storage', 'clean technology', 'green technology'
        ],
        
        'circular_economy': [
            # Waste & Circular Economy
            'kinh tế tuần hoàn', 'quản lý chất thải', 'tái chế', 'tái sử dụng', 'giảm thiểu chất thải',
            'xử lý chất thải', 'phân loại rác', 'ủ compost', 'biogas', 'zero waste',
            'giảm-tái sử dụng-tái chế', '3R', 'bao bì sinh học', 'bao bì tái chế', 'nhựa sinh học',
            'thiết kế xanh', 'sản phẩm xanh', 'chu trình sống sản phẩm', 'eco-design',
            'circular economy', 'waste management', 'recycling', 'upcycling', 'reuse',
            'waste reduction', 'zero waste', 'biodegradable', 'compostable',
            'life cycle assessment', 'LCA', 'sustainable design', 'green products'
        ],
        
        'water_stewardship': [
            # Water Management
            'quản lý nước', 'tiết kiệm nước', 'bảo vệ nguồn nước', 'chất lượng nước',
            'xử lý nước thải', 'tái sử dụng nước', 'hiệu quả sử dụng nước', 'bảo tồn nước',
            'nguồn tài nguyên nước', 'nước sạch', 'an toàn nước', 'thu gom nước mưa',
            'nước ngầm', 'ô nhiễm nước', 'giám sát chất lượng nước', 'tiêu chuẩn nước',
            'water management', 'water conservation', 'water efficiency', 'water quality',
            'water treatment', 'water recycling', 'water stewardship', 'clean water',
            'water safety', 'water security', 'water footprint', 'sustainable water'
        ],
        
        'biodiversity_nature': [
            # Biodiversity & Ecosystem
            'đa dạng sinh học', 'bảo vệ môi trường', 'bảo tồn thiên nhiên', 'hệ sinh thái',
            'môi trường sống', 'bảo vệ động vật hoang dã', 'bảo vệ thực vật', 'rừng bền vững',
            'trồng rừng', 'phục hồi rừng', 'chống phá rừng', 'khu bảo tồn', 'vườn quốc gia',
            'bảo vệ san hô', 'bảo vệ đại dương', 'phục hồi hệ sinh thái', 'dịch vụ hệ sinh thái',
            'biodiversity', 'ecosystem', 'conservation', 'wildlife protection', 'habitat',
            'sustainable forestry', 'deforestation', 'reforestation', 'marine conservation',
            'coral reef', 'ecosystem restoration', 'natural capital', 'ecosystem services'
        ],
        
        'sustainable_practices': [
            # Agriculture & Supply Chain
            'nông nghiệp bền vững', 'canh tác bền vững', 'chăn nuôi bền vững', 'nông nghiệp hữu cơ',
            'chuỗi cung ứng bền vững', 'nguồn cung bền vững', 'truy xuất nguồn gốc',
            'thương mại công bằng', 'công bằng trong thương mại', 'chứng nhận bền vững',
            'tiêu chuẩn môi trường', 'quản lý môi trường', 'ISO 14001', 'EMAS',
            'sustainable agriculture', 'organic farming', 'sustainable sourcing',
            'supply chain sustainability', 'traceability', 'fair trade', 'certification',
            'environmental management', 'sustainable procurement', 'green supply chain'
        ],
        
        'pollution_prevention': [
            # Pollution & Contamination
            'ô nhiễm không khí', 'ô nhiễm nước', 'ô nhiễm đất', 'chất độc hại',
            'kiểm soát ô nhiễm', 'giảm ô nhiễm', 'phòng ngừa ô nhiễm', 'làm sạch môi trường',
            'chất thải nguy hại', 'hóa chất độc', 'kim loại nặng', 'pesticide', 'phân bón hóa học',
            'tiếng ồn', 'ô nhiễm ánh sáng', 'bức xạ', 'vi nhựa', 'chất gây ung thư',
            'air pollution', 'water pollution', 'soil contamination', 'toxic substances',
            'pollution control', 'environmental remediation', 'hazardous waste',
            'chemical pollution', 'noise pollution', 'light pollution', 'microplastics'
        ]
    },
    
    'Social': {
        'workforce_development': [
            # Employee Relations & Development
            'phúc lợi nhân viên', 'quyền lợi nhân viên', 'chăm sóc nhân viên', 'phát triển nhân viên',
            'đào tạo và phát triển', 'nâng cao kỹ năng', 'học tập suốt đời', 'chuyển đổi số nhân sự',
            'cân bằng cuộc sống công việc', 'sức khỏe tinh thần', 'stress công việc', 'burnout',
            'môi trường làm việc tích cực', 'văn hóa doanh nghiệp', 'gắn kết nhân viên',
            'employee welfare', 'employee benefits', 'workforce development', 'skill development',
            'lifelong learning', 'digital transformation', 'work-life balance', 'mental health',
            'employee engagement', 'corporate culture', 'talent development', 'career growth'
        ],
        
        'health_safety': [
            # Occupational Health & Safety
            'an toàn lao động', 'sức khỏe nghề nghiệp', 'an toàn tại nơi làm việc',
            'phòng ngừa tai nạn', 'quản lý rủi ro an toàn', 'thiết bị bảo hộ', 'đào tạo an toàn',
            'khám sức khỏe định kỳ', 'chăm sóc sức khỏe', 'bảo hiểm y tế', 'hỗ trợ y tế',
            'sức khỏe cộng đồng', 'an toàn thực phẩm', 'vệ sinh công nghiệp', 'ergonomics',
            'occupational safety', 'workplace safety', 'health and safety', 'accident prevention',
            'safety management', 'safety training', 'health insurance', 'healthcare',
            'public health', 'food safety', 'industrial hygiene', 'occupational health'
        ],
        
        'diversity_inclusion': [
            # Diversity, Equity & Inclusion
            'đa dạng và hòa nhập', 'bình đẳng giới', 'bình đẳng cơ hội', 'không phân biệt đối xử',
            'đa dạng văn hóa', 'đa dạng thế hệ', 'đa dạng sắc tộc', 'đa dạng tôn giáo',
            'hòa nhập người khuyết tật', 'tiếp cận cho người khuyết tật', 'thiết kế toàn diện',
            'lãnh đạo đa dạng', 'cơ hội thăng tiến công bằng', 'lương bình đẳng', 'gender pay gap',
            'diversity and inclusion', 'gender equality', 'equal opportunity', 'non-discrimination',
            'cultural diversity', 'ethnic diversity', 'age diversity', 'disability inclusion',
            'inclusive design', 'diverse leadership', 'pay equity', 'unconscious bias'
        ],
        
        'community_engagement': [
            # Community Development & Social Impact
            'phát triển cộng đồng', 'đầu tư vào cộng đồng', 'tác động xã hội', 'trách nhiệm xã hội',
            'hoạt động từ thiện', 'tình nguyện viên', 'hỗ trợ giáo dục', 'học bổng',
            'xóa đói giảm nghèo', 'phát triển kinh tế địa phương', 'tạo việc làm', 'doanh nghiệp xã hội',
            'đối tác cộng đồng', 'tham gia cộng đồng', 'lắng nghe cộng đồng', 'phản hồi cộng đồng',
            'community development', 'social impact', 'community investment', 'social responsibility',
            'philanthropy', 'volunteering', 'education support', 'poverty alleviation',
            'local economic development', 'job creation', 'social enterprise', 'community partnership'
        ],
        
        'human_rights': [
            # Human Rights & Labor Standards
            'quyền con người', 'quyền cơ bản', 'quyền lao động', 'tiêu chuẩn lao động',
            'tự do kết hội', 'quyền thương lượng tập thể', 'không lao động cưỡng bức',
            'không lao động trẻ em', 'tuổi lao động tối thiểu', 'điều kiện làm việc nhân đạo',
            'tôn trọng nhân phẩm', 'công bằng và bình đẳng', 'quyền riêng tư', 'tự do ngôn luận',
            'human rights', 'fundamental rights', 'labor rights', 'labor standards',
            'freedom of association', 'collective bargaining', 'forced labor', 'child labor',
            'decent work', 'fair treatment', 'dignity', 'privacy rights', 'freedom of speech'
        ],
        
        'customer_stakeholder': [
            # Customer Relations & Stakeholder Engagement
            'trải nghiệm khách hàng', 'hài lòng khách hàng', 'chất lượng dịch vụ', 'dịch vụ khách hàng',
            'an toàn sản phẩm', 'chất lượng sản phẩm', 'trách nhiệm sản phẩm', 'bảo vệ người tiêu dùng',
            'quyền khách hàng', 'minh bạch thông tin sản phẩm', 'marketing có trách nhiệm',
            'tiếp cận công bằng', 'giá cả hợp lý', 'bảo vệ dữ liệu khách hàng', 'quyền riêng tư',
            'customer experience', 'customer satisfaction', 'service quality', 'product safety',
            'product quality', 'consumer protection', 'customer rights', 'responsible marketing',
            'fair access', 'fair pricing', 'data protection', 'privacy protection'
        ],
        
        'financial_inclusion': [
            # Financial Access & Inclusion
            'tài chính toàn diện', 'tiếp cận tài chính', 'bao trùm tài chính', 'dịch vụ tài chính cơ bản',
            'giáo dục tài chính', 'hiểu biết tài chính', 'tín dụng vi mô', 'cho vay có trách nhiệm',
            'ngân hàng số', 'thanh toán số', 'fintech', 'công nghệ tài chính', 'blockchain',
            'dịch vụ tài chính xanh', 'đầu tư có tác động', 'tài chính bền vững', 'ESG investing',
            'financial inclusion', 'financial access', 'financial literacy', 'microfinance',
            'responsible lending', 'digital banking', 'digital payments', 'fintech',
            'green finance', 'sustainable finance', 'impact investing', 'ESG investing'
        ]
    },
    
    'Governance': {
        'corporate_governance': [
            # Board & Management Structure
            'quản trị công ty', 'hội đồng quản trị', 'ban giám đốc', 'ban điều hành',
            'cơ cấu quản trị', 'thành viên độc lập', 'ban kiểm soát', 'ủy ban chuyên môn',
            'ủy ban kiểm toán', 'ủy ban nhân sự', 'ủy ban lương thưởng', 'ủy ban rủi ro',
            'đánh giá hiệu quả quản trị', 'quy trình ra quyết định', 'phân quyền và ủy quyền',
            'corporate governance', 'board of directors', 'executive management', 'independent directors',
            'board committees', 'audit committee', 'governance structure', 'decision making',
            'board effectiveness', 'management oversight', 'corporate structure'
        ],
        
        'ethics_integrity': [
            # Business Ethics & Anti-Corruption
            'đạo đức kinh doanh', 'liêm chính', 'minh bạch', 'trung thực', 'chính trực',
            'chống tham nhũng', 'chống hối lộ', 'xung đột lợi ích', 'quy tắc ứng xử',
            'văn hóa đạo đức', 'đường dây nóng đạo đức', 'tố giác vi phạm', 'whistleblowing',
            'quà tặng và tiếp khách', 'lợi ích cá nhân', 'giao dịch liên quan', 'fair dealing',
            'business ethics', 'integrity', 'transparency', 'anti-corruption', 'anti-bribery',
            'conflict of interest', 'code of conduct', 'ethical culture', 'whistleblowing',
            'gift policy', 'related party transactions', 'fair dealing', 'honest business'
        ],
        
        'risk_management': [
            # Enterprise Risk Management
            'quản lý rủi ro', 'quản lý rủi ro doanh nghiệp', 'đánh giá rủi ro', 'kiểm soát rủi ro',
            'giảm thiểu rủi ro', 'chuyển giao rủi ro', 'chấp nhận rủi ro', 'tránh rủi ro',
            'rủi ro chiến lược', 'rủi ro vận hành', 'rủi ro tài chính', 'rủi ro tuân thủ',
            'rủi ro danh tiếng', 'rủi ro công nghệ', 'rủi ro mạng', 'rủi ro khí hậu',
            'risk management', 'enterprise risk management', 'risk assessment', 'risk control',
            'risk mitigation', 'strategic risk', 'operational risk', 'financial risk',
            'compliance risk', 'reputational risk', 'cyber risk', 'climate risk'
        ],
        
        'compliance_legal': [
            # Regulatory Compliance & Legal
            'tuân thủ pháp luật', 'tuân thủ quy định', 'tuân thủ luật định', 'chính sách tuân thủ',
            'kiểm tra tuân thủ', 'giám sát tuân thủ', 'báo cáo tuân thủ', 'đảm bảo tuân thủ',
            'yêu cầu pháp lý', 'nghĩa vụ pháp lý', 'chế tài', 'vi phạm pháp luật',
            'chứng nhận', 'giấy phép', 'đăng ký', 'phê duyệt', 'kiểm định', 'kiểm tra',
            'legal compliance', 'regulatory compliance', 'compliance policy', 'compliance monitoring',
            'legal requirements', 'regulatory requirements', 'certification', 'licensing',
            'regulatory approval', 'audit', 'inspection', 'enforcement'
        ],
        
        'transparency_disclosure': [
            # Transparency & Reporting
            'minh bạch thông tin', 'công bố thông tin', 'báo cáo', 'công khai',
            'báo cáo thường niên', 'báo cáo bền vững', 'báo cáo ESG', 'báo cáo tài chính',
            'thông tin bắt buộc', 'thông tin tự nguyện', 'tiêu chuẩn báo cáo', 'chất lượng báo cáo',
            'kiểm toán độc lập', 'xác minh bên thứ ba', 'đảm bảo chất lượng', 'reliability',
            'transparency', 'disclosure', 'reporting', 'annual report', 'sustainability report',
            'ESG reporting', 'financial reporting', 'mandatory disclosure', 'voluntary disclosure',
            'reporting standards', 'independent audit', 'third party verification', 'assurance'
        ],
        
        'stakeholder_relations': [
            # Stakeholder Engagement & Communication
            'quan hệ bên liên quan', 'tương tác bên liên quan', 'giao tiếp bên liên quan',
            'quan hệ cổ đông', 'quan hệ nhà đầu tư', 'đối thoại với cổ đông', 'đại hội cổ đông',
            'lắng nghe ý kiến', 'thu thập phản hồi', 'tham vấn', 'đối thoại', 'hợp tác',
            'xây dựng niềm tin', 'uy tín', 'danh tiếng', 'thương hiệu', 'hình ảnh công ty',
            'stakeholder engagement', 'stakeholder relations', 'shareholder relations',
            'investor relations', 'stakeholder dialogue', 'feedback', 'consultation',
            'trust building', 'reputation', 'corporate image', 'brand reputation'
        ],
        
        'cybersecurity_data': [
            # Information Security & Data Protection
            'an ninh mạng', 'bảo mật thông tin', 'an toàn thông tin', 'bảo vệ dữ liệu',
            'quyền riêng tư', 'bảo mật dữ liệu cá nhân', 'GDPR', 'luật bảo vệ dữ liệu',
            'kiểm soát truy cập', 'xác thực', 'ủy quyền', 'mã hóa', 'firewall',
            'sao lưu dữ liệu', 'khôi phục thảm họa', 'liên tục hoạt động', 'ứng phó sự cố',
            'cybersecurity', 'information security', 'data protection', 'privacy',
            'data privacy', 'GDPR compliance', 'access control', 'authentication',
            'encryption', 'data backup', 'disaster recovery', 'business continuity'
        ],
        
        'innovation_technology': [
            # Digital Innovation & Technology Governance
            'đổi mới sáng tạo', 'chuyển đổi số', 'công nghệ mới', 'công nghệ số',
            'trí tuệ nhân tạo', 'AI', 'machine learning', 'big data', 'phân tích dữ liệu',
            'blockchain', 'IoT', 'cloud computing', 'tự động hóa', 'robotics',
            'quản trị công nghệ', 'quản trị dữ liệu', 'quản trị AI', 'đạo đức AI',
            'innovation', 'digital transformation', 'emerging technology', 'artificial intelligence',
            'machine learning', 'big data', 'data analytics', 'blockchain', 'IoT',
            'technology governance', 'data governance', 'AI governance', 'AI ethics'
        ]
    }
}

# Mapping categories to main ESG pillars for easy classification
esg_category_mapping = {
    'Environmental': [
        'climate_action', 'energy_transition', 'circular_economy', 
        'water_stewardship', 'biodiversity_nature', 'sustainable_practices', 
        'pollution_prevention'
    ],
    'Social': [
        'workforce_development', 'health_safety', 'diversity_inclusion',
        'community_engagement', 'human_rights', 'customer_stakeholder',
        'financial_inclusion'
    ],
    'Governance': [
        'corporate_governance', 'ethics_integrity', 'risk_management',
        'compliance_legal', 'transparency_disclosure', 'stakeholder_relations',
        'cybersecurity_data', 'innovation_technology'
    ]
}

total_keywords = sum(len(keywords) for category in esg_keywords.values() 
                    for keywords in category.values())

all_esg_keywords = []
for category in esg_keywords.values():
    for subcategory_keywords in category.values():
        all_esg_keywords.extend(subcategory_keywords)

# Chuyển tất cả từ khóa thành lowercase để so sánh
all_esg_keywords_lower = [keyword.lower() for keyword in all_esg_keywords]

# -----------------------------------------------------------------
# Analyze the Sentiment
#------------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import warnings
warnings.filterwarnings('ignore')

class FastSentimentRegressor(nn.Module):
    """
    Lightweight sentiment regression model using DistilBERT
    """
    def __init__(self, model_name='distilbert-base-multilingual-cased', dropout_rate=0.3):
        super(FastSentimentRegressor, self).__init__()
        
        # Load pre-trained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Regression head
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0-1
        )
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply dropout and regression head
        pooled_output = self.dropout(pooled_output)
        score = self.regressor(pooled_output)
        
        return score.squeeze()  # Remove last dimension

def load_sentiment_model(model_path='sentiment_regressor_complete.pth', device='cpu'):
    """
    Load the trained sentiment regression model
    
    Args:
        model_path (str): Path to the saved model file
        device (str): Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    try:
        print(f"📥 Loading sentiment model from {model_path}...")
        
        # Set device
        device = torch.device(device)
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Initialize model
        model = FastSentimentRegressor(
            model_name=config['model_name'],
            dropout_rate=config['dropout_rate']
        )
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(device)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

# 🚀 MAIN INFERENCE FUNCTION
def infer_sentiment(vietnamese_sentence):
    # Validate input
    if not isinstance(vietnamese_sentence, str):
        raise TypeError("Input must be a string")
    
    if len(vietnamese_sentence.strip()) == 0:
        raise ValueError("Input sentence cannot be empty")
    
    # Check if model is loaded
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Please run the model loading cell first.")
    
    try:
        with torch.no_grad():
            # Tokenize the sentence
            encoded = tokenizer(
                vietnamese_sentence,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            # Get prediction
            score = model(encoded['input_ids'], encoded['attention_mask'])
            
            # Return as Python float
            return float(score.item())
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise

print("🚀 Inference function defined!")

# ==============================
# DATAFRAME CREATION WITH ALL FEATURES
# ==============================

def extract_organization_names(text):
    """Processes text with the NER pipeline and returns a list of organization names."""
    if len(text) > 512:
        text = text[:512]

    ner_results = ner_pipeline(text)
    organization_names = []
    for entity in ner_results:

        if "ORG" in entity['entity_group'].upper(): # Case-insensitive check
            organization_names.append(entity['word'])
    return organization_names

def process_esg_files_working(texts: str, filename: str):
    all_results = []
    features = {
        'filename': filename,
        
        # Environmental features
        'pos_env_climate_action': 0, 'neg_env_climate_action': 0,
        'pos_env_energy_transition': 0, 'neg_env_energy_transition': 0,
        'pos_env_water_stewardship': 0, 'neg_env_water_stewardship': 0,
        'pos_env_biodiversity_nature': 0, 'neg_env_biodiversity_nature': 0,
        'pos_env_pollution_prevention': 0, 'neg_env_pollution_prevention': 0,
        'pos_env_circular_economy': 0, 'neg_env_circular_economy': 0,
        'pos_env_sustainable_practices': 0, 'neg_env_sustainable_practices': 0,
        
        # Social features
        'pos_social_diversity_inclusion': 0, 'neg_social_diversity_inclusion': 0,
        'pos_social_workforce_development': 0, 'neg_social_workforce_development': 0,
        'pos_social_health_safety': 0, 'neg_social_health_safety': 0,
        'pos_social_human_rights': 0, 'neg_social_human_rights': 0,
        'pos_social_community_engagement': 0, 'neg_social_community_engagement': 0,
        'pos_social_customer_stakeholder': 0, 'neg_social_customer_stakeholder': 0,
        'pos_social_financial_inclusion': 0, 'neg_social_financial_inclusion': 0,
        
        # Governance features
        'pos_gov_corporate_governance': 0, 'neg_gov_corporate_governance': 0,
        'pos_gov_ethics_integrity': 0, 'neg_gov_ethics_integrity': 0,
        'pos_gov_transparency_disclosure': 0, 'neg_gov_transparency_disclosure': 0,
        'pos_gov_risk_management': 0, 'neg_gov_risk_management': 0,
        'pos_gov_compliance_legal': 0, 'neg_gov_compliance_legal': 0,
        'pos_gov_stakeholder_relations': 0, 'neg_gov_stakeholder_relations': 0,
        'pos_gov_innovation_technology': 0, 'neg_gov_innovation_technology': 0,
        'pos_gov_cybersecurity_data': 0, 'neg_gov_cybersecurity_data': 0,
    }
    
    try:
        sentences = re.split(r'[.!?]+', texts)
        features['total_sentences'] = len(sentences)
        features['total_words'] = len(texts.split())
        features['NER_pos'] = 0
        features['NER_neg'] = 0

        esg_sentences = []
        esg_count = 0
        pos_count = 0
        neg_count = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            sentence_lower = sentence.lower()
            
            # Find ESG keywords
            found_keywords = []
            for keyword in all_esg_keywords_lower:
                if keyword in sentence_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                esg_count += 1
                
                # Find categories and subcategories
                categories_found = set()
                subcategories_found = set()
                
                for category, subcategories in esg_keywords.items():
                    for subcategory, keywords in subcategories.items():
                        for keyword in keywords:
                            if keyword.lower() in sentence_lower:
                                categories_found.add(category)
                                subcategories_found.add(subcategory)
                
                if (len(sentence.split()) > 50):
                    sentence = ' '.join(sentence.split()[:50])
                # Sentiment analysis
                sentiment_score = infer_sentiment(sentence)
                sentiment_label = 'neutral'
                

                if sentiment_score >= 0.7:
                    sentiment_label = 'positive'
                elif sentiment_score < 0.5:
                    sentiment_label = 'negative'

                name_list = extract_organization_names(sentence)
                for name in name_list:
                    if name.lower() not in company_esg_dict:
                        continue
                    else:
                        point = company_esg_dict[name.lower()]
                        if point < 0:
                            features['NER_neg'] += abs(point)
                        else:
                            features['NER_pos'] += abs(point)

                confidence = sentiment_score if sentiment_label == 'positive' else (1 - sentiment_score)
                
                esg_sentence_data = {
                    'sentence_id': i,
                    'sentence': sentence,
                    'keywords_found': found_keywords,
                    'categories': list(categories_found),
                    'subcategories': list(subcategories_found),
                    'keyword_count': len(found_keywords),
                    'sentiment': sentiment_label,
                    'confidence': confidence
                }
                
                # Count features based on sentiment and subcategory
                if sentiment_label == 'positive':
                    pos_count += 1
                    for subcategory in subcategories_found:
                        # Map subcategory to feature name and increment
                        if subcategory == 'climate_action':
                            features['pos_env_climate_action'] += 1
                        elif subcategory == 'energy_transition':
                            features['pos_env_energy_transition'] += 1
                        elif subcategory == 'water_stewardship':
                            features['pos_env_water_stewardship'] += 1
                        elif subcategory == 'biodiversity_nature':
                            features['pos_env_biodiversity_nature'] += 1
                        elif subcategory == 'pollution_prevention':
                            features['pos_env_pollution_prevention'] += 1
                        elif subcategory == 'circular_economy':
                            features['pos_env_circular_economy'] += 1
                        elif subcategory == 'sustainable_practices':
                            features['pos_env_sustainable_practices'] += 1
                        elif subcategory == 'diversity_inclusion':
                            features['pos_social_diversity_inclusion'] += 1
                        elif subcategory == 'workforce_development':
                            features['pos_social_workforce_development'] += 1
                        elif subcategory == 'health_safety':
                            features['pos_social_health_safety'] += 1
                        elif subcategory == 'human_rights':
                            features['pos_social_human_rights'] += 1
                        elif subcategory == 'community_engagement':
                            features['pos_social_community_engagement'] += 1
                        elif subcategory == 'customer_stakeholder':
                            features['pos_social_customer_stakeholder'] += 1
                        elif subcategory == 'financial_inclusion':
                            features['pos_social_financial_inclusion'] += 1
                        elif subcategory == 'corporate_governance':
                            features['pos_gov_corporate_governance'] += 1
                        elif subcategory == 'ethics_integrity':
                            features['pos_gov_ethics_integrity'] += 1
                        elif subcategory == 'transparency_disclosure':
                            features['pos_gov_transparency_disclosure'] += 1
                        elif subcategory == 'risk_management':
                            features['pos_gov_risk_management'] += 1
                        elif subcategory == 'compliance_legal':
                            features['pos_gov_compliance_legal'] += 1
                        elif subcategory == 'stakeholder_relations':
                            features['pos_gov_stakeholder_relations'] += 1
                        elif subcategory == 'innovation_technology':
                            features['pos_gov_innovation_technology'] += 1
                        elif subcategory == 'cybersecurity_data':
                            features['pos_gov_cybersecurity_data'] += 1
                            
                elif sentiment_label == 'negative':
                    neg_count += 1
                    for subcategory in subcategories_found:
                        # Map subcategory to feature name and increment
                        if subcategory == 'climate_action':
                            features['neg_env_climate_action'] += 1
                        elif subcategory == 'energy_transition':
                            features['neg_env_energy_transition'] += 1
                        elif subcategory == 'water_stewardship':
                            features['neg_env_water_stewardship'] += 1
                        elif subcategory == 'biodiversity_nature':
                            features['neg_env_biodiversity_nature'] += 1
                        elif subcategory == 'pollution_prevention':
                            features['neg_env_pollution_prevention'] += 1
                        elif subcategory == 'circular_economy':
                            features['neg_env_circular_economy'] += 1
                        elif subcategory == 'sustainable_practices':
                            features['neg_env_sustainable_practices'] += 1
                        elif subcategory == 'diversity_inclusion':
                            features['neg_social_diversity_inclusion'] += 1
                        elif subcategory == 'workforce_development':
                            features['neg_social_workforce_development'] += 1
                        elif subcategory == 'health_safety':
                            features['neg_social_health_safety'] += 1
                        elif subcategory == 'human_rights':
                            features['neg_social_human_rights'] += 1
                        elif subcategory == 'community_engagement':
                            features['neg_social_community_engagement'] += 1
                        elif subcategory == 'customer_stakeholder':
                            features['neg_social_customer_stakeholder'] += 1
                        elif subcategory == 'financial_inclusion':
                            features['neg_social_financial_inclusion'] += 1
                        elif subcategory == 'corporate_governance':
                            features['neg_gov_corporate_governance'] += 1
                        elif subcategory == 'ethics_integrity':
                            features['neg_gov_ethics_integrity'] += 1
                        elif subcategory == 'transparency_disclosure':
                            features['neg_gov_transparency_disclosure'] += 1
                        elif subcategory == 'risk_management':
                            features['neg_gov_risk_management'] += 1
                        elif subcategory == 'compliance_legal':
                            features['neg_gov_compliance_legal'] += 1
                        elif subcategory == 'stakeholder_relations':
                            features['neg_gov_stakeholder_relations'] += 1
                        elif subcategory == 'innovation_technology':
                            features['neg_gov_innovation_technology'] += 1
                        elif subcategory == 'cybersecurity_data':
                            features['neg_gov_cybersecurity_data'] += 1
                
                esg_sentences.append(esg_sentence_data)
        
        # Calculate aggregated features
        env_pos = sum([features[f'pos_env_{sub}'] for sub in ['climate_action', 'energy_transition', 'water_stewardship', 'biodiversity_nature', 'pollution_prevention', 'circular_economy', 'sustainable_practices']])
        env_neg = sum([features[f'neg_env_{sub}'] for sub in ['climate_action', 'energy_transition', 'water_stewardship', 'biodiversity_nature', 'pollution_prevention', 'circular_economy', 'sustainable_practices']])
        social_pos = sum([features[f'pos_social_{sub}'] for sub in ['diversity_inclusion', 'workforce_development', 'health_safety', 'human_rights', 'community_engagement', 'customer_stakeholder', 'financial_inclusion']])
        social_neg = sum([features[f'neg_social_{sub}'] for sub in ['diversity_inclusion', 'workforce_development', 'health_safety', 'human_rights', 'community_engagement', 'customer_stakeholder', 'financial_inclusion']])
        gov_pos = sum([features[f'pos_gov_{sub}'] for sub in ['corporate_governance', 'ethics_integrity', 'transparency_disclosure', 'risk_management', 'compliance_legal', 'stakeholder_relations', 'innovation_technology', 'cybersecurity_data']])
        gov_neg = sum([features[f'neg_gov_{sub}'] for sub in ['corporate_governance', 'ethics_integrity', 'transparency_disclosure', 'risk_management', 'compliance_legal', 'stakeholder_relations', 'innovation_technology', 'cybersecurity_data']])
        
        # Add aggregated features
        features.update({
            'total_pos_environmental': env_pos,
            'total_neg_environmental': env_neg,
            'total_pos_social': social_pos,
            'total_neg_social': social_neg,
            'total_pos_governance': gov_pos,
            'total_neg_governance': gov_neg,
            'total_environmental_mentions': env_pos + env_neg,
            'total_social_mentions': social_pos + social_neg,
            'total_governance_mentions': gov_pos + gov_neg,
            'total_esg_mentions': env_pos + env_neg + social_pos + social_neg + gov_pos + gov_neg,
            'esg_pos_ratio': (env_pos + social_pos + gov_pos) / max(env_pos + env_neg + social_pos + social_neg + gov_pos + gov_neg, 1),
            'esg_neg_ratio': (env_neg + social_neg + gov_neg) / max(env_pos + env_neg + social_pos + social_neg + gov_pos + gov_neg, 1),
        })
        
        all_results.append(features)
        
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

    # Create final dataframe
    df_all_files = pd.DataFrame(all_results) if all_results else None
    
    if df_all_files is not None:
        print(f"\\n📊 THÀNH CÔNG!")
    
    return df_all_files

def assign_cluster(df_new, feature_cols, cluster_centroids, scaler):
    df_new_scaled = scaler.transform(df_new[feature_cols])
    
    min_dist = float('inf')
    assigned_cluster = None
    
    for cluster, centroid in cluster_centroids.items():
        centroid_scaled = scaler.transform(centroid.values.reshape(1, -1))
        
        dist = np.linalg.norm(df_new_scaled - centroid_scaled)
        
        if dist < min_dist:
            min_dist = dist
            assigned_cluster = cluster
    
    return assigned_cluster

def infer_esg_scores(df, model_path='d:/Jupyter/hackathon_techcombank/'):
    """
    Inference function to predict E, S, G scores from input dataframe
    
    Args:
        df: pandas DataFrame with features (without labels)
        model_path: Path to the saved models directory
    
    Returns:
        pandas DataFrame with columns ['e_score', 's_score', 'g_score']
    """
    import pandas as pd
    import numpy as np
    import joblib
    import xgboost as xgb
    
    print("=== ESG SCORE INFERENCE ===")
    print(f"Input data shape: {df.shape}")
    
    # Load saved models and preprocessing objects
    try:
        e_model = joblib.load(f'{model_path}xgboost_e_score_model.pkl')
        s_model = joblib.load(f'{model_path}xgboost_s_score_model.pkl')
        g_model = joblib.load(f'{model_path}xgboost_g_score_model.pkl')
        scaler = joblib.load(f'{model_path}xgboost_scaler.pkl')
        label_encoders = joblib.load(f'{model_path}xgboost_encoders.pkl')
        feature_names = joblib.load(f'{model_path}xgboost_features.pkl')
        
        print("Models loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please run train_esg_models() first to train and save the models.")
        return None
    
    # Prepare input data
    X = df.copy()
    
    # Remove non-feature columns if they exist
    exclude_cols = ['company_name', 'symbol', 'e_score', 's_score', 'g_score']
    for col in exclude_cols:
        if col in X.columns:
            X = X.drop(columns=[col])
    
    # Handle categorical variables using saved encoders
    for col, encoder in label_encoders.items():
        if col in X.columns:
            try:
                X[col] = encoder.transform(X[col].astype(str))
            except ValueError:
                # Handle unseen categories by assigning a default value
                print(f"Warning: Unseen categories in column '{col}', using most frequent class")
                X[col] = X[col].astype(str)
                # Replace unseen categories with the most frequent category from training
                known_classes = set(encoder.classes_)
                X[col] = X[col].apply(lambda x: x if x in known_classes else encoder.classes_[0])
                X[col] = encoder.transform(X[col])
    
    # Ensure we have all required features
    missing_features = set(feature_names) - set(X.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with default values (median or 0)
        for feature in missing_features:
            X[feature] = 0
    
    # Select and reorder features to match training
    X = X[feature_names]
    
    # Handle missing values (fill with median of available data)
    X = X.fillna(X.median())
    
    # If still any missing values, fill with 0
    X = X.fillna(0)
    
    print(f"Processed features shape: {X.shape}")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    e_scores = e_model.predict(X_scaled)
    s_scores = s_model.predict(X_scaled)
    g_scores = g_model.predict(X_scaled)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'e_score': e_scores,
        's_score': s_scores,
        'g_score': g_scores
    })
    
    print("=== INFERENCE COMPLETED ===")
    print(f"Predicted scores shape: {results_df.shape}")
    
    return results_df

if __name__ == "__main__":
    stored_text = pdf_to_text('D:/Jupyter/hackathon_techcombank/esg_report_pdf/AR SAB 2023.pdf')
    
    model, tokenizer, device = load_sentiment_model()
    model_name = "NlpHUST/ner-vietnamese-electra-base" 

    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_ner = AutoModelForTokenClassification.from_pretrained(model_name)

    ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer, device='cpu', grouped_entities=True)

    df = pd.read_csv('company_esg.csv')

    import re

    def clean_company_name(name):
        prefixes = ['Công ty CP', 'Công ty Cổ phần', 'Công ty TNHH', 'Tập đoàn', 'Ngân hàng TMCP', 'Ngân hàng', 'Công ty']
        prefixes.sort(key=len, reverse=True)
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):].strip()
                break # Remove only one prefix
        return name

    df['company_name'] = df['company_name'].apply(clean_company_name)

    company_esg_dict = {}
    for index, row in df.iterrows():
        company_esg_dict[row['company_name'].lower()] = row[' esg_score'] - 2.5

    df_all_files = process_esg_files_working(stored_text, 'filename')

    df_train = pd.read_csv('esg_features_with_ner_scores.csv')

    feature_cols = [col for col in df_train.columns if col not in 
                ['filename', 'esg_tier', 'esg_cluster', 'e_score', 's_score', 'g_score']]

    cluster_centroids = {}
    for cluster in df_train['esg_cluster'].unique():
        cluster_data = df_train[df_train['esg_cluster'] == cluster]
        cluster_centroids[cluster] = cluster_data[feature_cols].mean()

    scaler = StandardScaler()
    scaler.fit(df_train[feature_cols])

    assigned_cluster = assign_cluster(df_all_files, feature_cols, cluster_centroids, scaler)

    df_all_files['esg_cluster'] = assigned_cluster

    # this has the output of 20 features
    df_all_files.to_csv('esg_features_bbc_2023.csv', index=False)

    inferred_scores = infer_esg_scores(df_all_files, model_path='d:/Jupyter/hackathon_techcombank/')

    print(inferred_scores) # This is the return score (E, S, G)

