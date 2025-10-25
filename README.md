# Marketing Brand Intelligence (MBI) System

A comprehensive AI-driven marketing intelligence and CRM platform combining Marketing Mix Modeling (MMM), Multi-Touch Attribution (MTA), AI agents, and Salesforce-equivalent CRM capabilities.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+

### Local Development Setup

1. **Clone and navigate to project**
```bash
cd C:\Users\ishop\OneDrive\Documents\GitHub\MBI
```

2. **Set up Python virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
copy .env.example .env
# Edit .env with your configuration
```

4. **Start infrastructure services**
```bash
docker-compose up -d postgres redis kafka
```

5. **Run database migrations**
```bash
alembic upgrade head
```

6. **Start backend server**
```bash
uvicorn app.main:app --reload --port 8000
```

7. **Start frontend (in new terminal)**
```bash
cd frontend
npm install
npm run dev
```

8. **Access the application**
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs
- API: http://localhost:8000/api/v2

## 📁 Project Structure

```
MBI/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── agents/            # AI agents (MMM, MTA, Lead Scoring, etc.)
│   │   ├── api/               # API routes
│   │   ├── core/              # Core utilities, config, security
│   │   ├── crm/               # CRM models and services
│   │   ├── db/                # Database models and migrations
│   │   ├── intelligence/      # ML models and algorithms
│   │   ├── llm/               # LLM integration (RAG, prompts)
│   │   ├── playbooks/         # Automation playbooks
│   │   └── main.py            # Application entry point
│   ├── tests/                 # Backend tests
│   └── requirements.txt
├── frontend/                   # Next.js 14 frontend
│   ├── app/                   # App router pages
│   ├── components/            # React components
│   ├── lib/                   # Utilities and API client
│   └── package.json
├── infrastructure/             # Terraform IaC
│   ├── gcp/
│   └── aws/
├── dbt/                       # Data transformation
│   ├── models/
│   └── dbt_project.yml
├── docker/                    # Docker configurations
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── docker-compose.yml
├── .env.example
├── AGENT.md                   # Full system architecture
└── BOOTSTRAPPER.md            # Setup guide
```

## 🎯 Core Features

### Marketing Intelligence
- **MMM (Marketing Mix Modeling)**: Bayesian attribution with PyMC
- **MTA (Multi-Touch Attribution)**: Markov chains & Shapley values
- **Brand Tracking**: Share of Search, Share of Voice, sentiment analysis
- **Creative Intelligence**: AI-powered asset analysis and variant generation
- **Audience Expansion**: Lookalike modeling and automated A/B testing

### CRM System
- **Leads**: Scoring, routing, nurturing
- **Accounts**: Company management, health scoring
- **Contacts**: Relationship mapping
- **Opportunities**: Pipeline management, forecasting
- **Cases**: Customer support ticketing
- **Activities**: Task, email, call tracking

### AI Agents (15+)
- Identity Resolution Agent
- Lead Scoring Agent (XGBoost + Neural Net ensemble)
- MMM Agent (Bayesian with adstock/saturation)
- MTA Agent (Markov/Shapley)
- Creative Intelligence Agent (CLIP + fatigue detection)
- Crisis Detection Agent (LLM + multi-source verification)
- Budget Allocation Agent (constrained optimization)
- Audience Expansion Agent (lookalike + A/B testing)
- And 7+ more specialized agents

### Automation Playbooks
- Budget reallocation (weekly optimization)
- Creative rotation (fatigue detection)
- Audience expansion (lookalike + testing)
- Crisis response (brand protection)
- Lead nurturing (multi-touch sequences)
- Opportunity follow-up (SLA enforcement)

## 🔧 Technology Stack

### Backend
- **Framework**: FastAPI 0.104+
- **Language**: Python 3.11+
- **Database**: PostgreSQL 15+ (operational), BigQuery (analytics)
- **Cache**: Redis 7+
- **ML**: PyMC 5.0+, scikit-learn, TensorFlow
- **LLM**: Anthropic Claude, OpenAI GPT
- **Workflow**: Prefect, Celery
- **Data**: dbt, pandas, polars

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript 5.3+
- **Styling**: Tailwind CSS + shadcn/ui
- **State**: Zustand + TanStack Query
- **Charts**: Recharts, D3.js
- **Forms**: React Hook Form + Zod

### Infrastructure
- **Container**: Docker, Kubernetes
- **IaC**: Terraform (GCP/AWS)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Logging**: Loki / ELK Stack

## 📊 Key Metrics & KPIs

### Brand Health
- Share of Search (SoS)
- Share of Voice (SOV)
- Brand Lift
- Sentiment Score

### Performance
- ROAS (Return on Ad Spend) - Target: ≥2.5x
- CAC (Customer Acquisition Cost)
- LTV/CAC Ratio - Target: ≥3:1
- Conversion Rate by Channel

### Operational
- API Response Time: <200ms (p95)
- Data Freshness: <6 hours lag
- Model Accuracy: MAPE <15% (MMM)
- Automation Rate: ≥60% of decisions

## 🔐 Security & Compliance

- **Privacy**: Hash PII immediately (SHA256 + salt)
- **GDPR**: Auto-delete after 90 days, right to be forgotten
- **RBAC**: Role-based access control for all resources
- **Audit**: Full decision trail logging
- **Encryption**: At rest and in transit
- **LLM**: RAG-only, no PII exposure

## 🧪 Testing

```bash
# Backend tests
pytest tests/ -v --cov=app

# Frontend tests
cd frontend
npm test

# Integration tests
pytest tests/integration/ -v

# Load tests
locust -f tests/load/locustfile.py
```

## 📖 Documentation

- [System Architecture](./AGENT.md) - Complete technical specification
- [Bootstrapper Guide](./BOOTSTRAPPER.md) - Development setup guide
- [API Reference](http://localhost:8000/docs) - OpenAPI documentation
- [Agent Specifications](./docs/agents/) - Individual agent details
- [Playbook Library](./docs/playbooks/) - Automation workflows

## 🚀 Deployment

### Development
```bash
docker-compose up
```

### Staging/Production
```bash
# Using Terraform
cd infrastructure/gcp  # or aws
terraform init
terraform plan
terraform apply

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 📈 Roadmap

### MVP (Weeks 1-2) ✅
- [x] Data ingestion (Meta, Google, GA4, Shopify)
- [x] Basic MMM & MTA
- [x] Creative tagging
- [x] 2 core playbooks

### Phase 2 (Weeks 3-4)
- [ ] LLM Council integration
- [ ] Crisis detection
- [ ] Advanced attribution
- [ ] CRM core (Leads, Accounts, Opportunities)

### Phase 3 (Weeks 5-8)
- [ ] Full CRM features
- [ ] Advanced playbooks
- [ ] Mobile app (React Native)
- [ ] Real-time optimization

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Follow code standards (black, mypy, eslint)
3. Write tests (min 80% coverage)
4. Submit PR with clear description

## 📄 License

Internal use only - Proprietary

## 🆘 Support

- Issues: [GitHub Issues](https://github.com/company/mbi/issues)
- Docs: [Internal Wiki](https://wiki.company.com/mbi)
- Slack: #mbi-support

---

**Built with:** Python • FastAPI • Next.js • PostgreSQL • Redis • PyMC • Claude API • Kubernetes

**Version:** 2.1.0
