#pragma once

#include <map>
#include <string>
#include <list>

class Args
{
public:
	using string = std::string;
	using MapStrStr = std::map<string, string>;
	using ListStr = std::list<string>;

	Args(int argc, char* argv[], const MapStrStr& mapDescription = {}, const string& sep = "-") {
		m_mapDescription = mapDescription;

		string strKey;
		for (int n = 1; n < argc; n++) {
			string str = escape(argv[n]);
			if (str.find(sep) == 0) {
				if (!strKey.empty()) {
					m_mapParam.emplace(strKey, "");
				}
				strKey = str.substr(0 + sep.size());
				std::transform(strKey.begin(), strKey.end(), strKey.begin(), ::tolower);
			}
			else if (!strKey.empty()) {
				m_mapParam.emplace(strKey, str);
				strKey.clear();
			}
		}
		if (!strKey.empty()) {
			m_mapParam.emplace(strKey, "");
		}
	}

	bool HasKey(string strKey) {
		std::transform(strKey.begin(), strKey.end(), strKey.begin(), ::tolower);
		return m_mapParam.find(strKey) != m_mapParam.end();
	}
	

	string GetValue(string strKeys, string strDefValue = "") {  //strKeys 以空格分割的参数
		ListStr keys;
		splitKey(strKeys, keys);

		for (auto key : keys) {
			string strVal;
			if (getValue(key, strVal)) {
				return strVal;
			}
		}
		return strDefValue;
	}

	string GetInvalidKey() {

		ListStr keys;
		for (auto desp : m_mapDescription) {
			splitKey(desp.first, keys);
		}

		if (m_mapParam.empty() || keys.empty()) {
			return "";
		}

		for (auto param : m_mapParam) {
			if (std::find(keys.begin(), keys.end(), param.first) == keys.end()) {
				return param.first;
			}
		}
		return "";
	}

	string Useage() {
		static string  strUag;

		if (strUag.empty()) {
			strUag = "[Useage]\n";

			for (auto desp : m_mapDescription) {
				ListStr keys;
				splitKey(desp.first, keys);

				int n = 0;
				for (auto key : keys) {
					if (n != 0) strUag += ",";
					strUag += "-";
					strUag += key;
					n++;
				}
				strUag += ": \t-- ";
				strUag += desp.second;
				strUag += ".\n";
			}
		}

		return strUag;
	}

protected:
	void splitKey(const string& strKeys, ListStr& keys) {
		string strKey;
		for (int n = 0; n < strKeys.size(); n++) {
			char c = strKeys[n];
			if (!isalnum(c)) {
				if (!strKey.empty()) {
					keys.push_back(strKey);
				}
				strKey.clear();
			}
			else {
				strKey.push_back(c);
			}
		}
		if (!strKey.empty()) {
			keys.push_back(strKey);
		}
	}

	bool getValue(string strKey, string& strValue) {
		std::transform(strKey.begin(), strKey.end(), strKey.begin(), ::tolower);
		MapStrStr::iterator it = m_mapParam.find(strKey);
		if (it != m_mapParam.end()) {
			strValue = it->second;
			return true;
		}
		return false;
	}

	string escape(string s) {
		string ret;
		for (int n = 0; n < s.size(); n++) {
			if (!isspace(s[n])) {
				ret.push_back(s[n]);
			}
		}
		return ret;
	}

private:
	MapStrStr m_mapParam;
	MapStrStr m_mapDescription;
};
