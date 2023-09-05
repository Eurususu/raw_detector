#pragma once

#include <string>
#include <list>
#include <memory>

namespace AQT {

	class Dir
	{
	public:
		using string = std::string;
		using ListStr = std::list<string>;
		using ptrDir = std::shared_ptr<Dir>;

		Dir(const string& strDir) {
			string dir = escape(strDir);
			if (!dir.empty()) {
				init(dir);
			}
		}

		string Name() {
			return m_node;
		}

		string FullName() {
			if (!m_parent) {
				return m_node;
			}
			return m_parent->FullName() + "/" + m_node;
		}

		ptrDir Parent() {
			return m_parent;
		}

		string Root() {
			if (!m_parent) {
				return m_node;
			}
			return m_parent->Root();
		}

	protected:
		Dir(const string& dir, int priv) {
			init(dir);
		}

		void init(const string& dir) {
			for (int n = dir.size() - 1; n >= 0; n--) {
				if (isSep(dir[n])) {
					m_node = dir.substr(n + 1);
					string strDir = dir.substr(0, n);
					m_parent.reset(new Dir(strDir, 0));
					break;
				}
			}
		}

		bool isSep(char c) {
			return c == '/' || c == '\\';
		}
		string escape(const string& dir) {
			string str;
			bool bSep = false;
			for (int n = 0; n < dir.size(); n++) {
				char c = dir[n];
				bool b = isSep(c);
				if (!b || !bSep) str.push_back(c);
				bSep = b;
			}
			if (!str.empty() && isSep(str[str.size() - 1])) {
				str[str.size() - 1] = 0;
				str = str.c_str();
			}
			return str;
		}
	private:
		ptrDir m_parent;
		string m_node;
	};
};